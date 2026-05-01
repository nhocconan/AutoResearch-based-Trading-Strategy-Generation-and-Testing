#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA with RSI filter and volume spike. Long when KAMA rising AND RSI(14) > 50 AND volume > 1.8x 20-period average.
# Short when KAMA falling AND RSI(14) < 50 AND volume confirmation. Uses discrete sizing 0.25. ATR stoploss: signal→0 when price moves against position by 2.5*ATR.
# KAMA adapts to market noise, reducing whipsaw in ranging markets. Volume spike ensures momentum confirmation.
# Works in bull (rising KAMA with RSI>50) and bear (falling KAMA with RSI<50). Target: 20-40 trades/year on 1d timeframe.

name = "1d_KAMA_RSI_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Load 1d data ONCE before loop for KAMA, RSI, and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def kama(source, period=10, fast=2, slow=30):
        if len(source) < period:
            return np.full_like(source, np.nan, dtype=float)
        change = np.abs(np.diff(source, n=period))
        volatility = np.sum(np.abs(np.diff(source)), axis=0) if len(source) > 1 else np.array([0])
        volatility = pd.Series(volatility).rolling(window=period, min_periods=1).sum().values
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama_vals = np.full_like(source, np.nan, dtype=float)
        kama_vals[period-1] = source[period-1]
        for i in range(period, len(source)):
            if not np.isnan(sc[i]) and not np.isnan(kama_vals[i-1]):
                kama_vals[i] = kama_vals[i-1] + sc[i] * (source[i] - kama_vals[i-1])
            else:
                kama_vals[i] = np.nan
        return kama_vals
    
    kama_vals = kama(close_1d, 10, 2, 30)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_vals)
    
    # Calculate RSI(14) on 1d
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan], rsi])  # align length
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 1d volume average (20-period)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, KAMA, RSI, and volume
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume spike: current volume > 1.8x 1d volume average
        if vol_ma_1d_aligned[i] <= 0 or np.isnan(vol_ma_1d_aligned[i]):
            volume_spike = False
        else:
            volume_spike = curr_volume > (vol_ma_1d_aligned[i] * 1.8)
        
        # KAMA direction: rising if current > previous, falling if current < previous
        kama_rising = kama_aligned[i] > kama_aligned[i-1]
        kama_falling = kama_aligned[i] < kama_aligned[i-1]
        
        # RSI filter
        rsi_above_50 = rsi_aligned[i] > 50
        rsi_below_50 = rsi_aligned[i] < 50
        
        if position == 0:  # Flat - look for new entries
            # Long: KAMA rising AND RSI > 50 AND volume spike
            if kama_rising and rsi_above_50 and volume_spike:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: KAMA falling AND RSI < 50 AND volume spike
            elif kama_falling and rsi_below_50 and volume_spike:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: KAMA starts falling OR RSI drops below 50
            elif not kama_rising or rsi_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: KAMA starts rising OR RSI rises above 50
            elif not kama_falling or rsi_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals