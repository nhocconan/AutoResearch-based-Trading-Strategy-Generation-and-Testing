#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 12h EMA100 trend filter and volume confirmation.
# Long when price breaks above Camarilla R4 AND close > 12h EMA100 AND volume > 1.5x 4h volume median.
# Short when price breaks below Camarilla S4 AND close < 12h EMA100 AND volume > 1.5x 4h volume median.
# Uses discrete sizing 0.25. ATR(10) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Camarilla R4/S4 are stronger intraday levels (1.1/2 multiplier) than R3/S3, reducing false breakouts.
# 12h EMA100 provides a slower, more reliable trend filter than EMA50.
# Volume confirmation at 1.5x median (vs 2.0x) balances sensitivity with noise reduction.
# Target: 30-60 trades/year on 4h timeframe to avoid fee drag while capturing meaningful moves.
# Proven pattern: tighter entry conditions + volume + slower trend filter works on BTC/ETH in both bull/bear.

name = "4h_Camarilla_R4S4_Breakout_12hEMA100_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(10) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 4h volume median (20-period for stability)
    vol_median_4h = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate 12h EMA100 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 100:
        return np.zeros(n)
    
    ema_100_12h = pd.Series(df_12h['close'].values).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_100_12h)
    
    # Calculate Camarilla levels from prior 1d bar
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla: based on prior day's high, low, close
    h1 = df_1d['high'].shift(1).values  # prior day high
    l1 = df_1d['low'].shift(1).values   # prior day low
    c1 = df_1d['close'].shift(1).values # prior day close
    
    # Calculate Camarilla R4 and S4 levels (stronger levels: 1.1/2 multiplier)
    # R4 = c1 + (h1 - l1) * 1.1/2
    # S4 = c1 - (h1 - l1) * 1.1/2
    camarilla_range = h1 - l1
    r4 = c1 + camarilla_range * 1.1 / 2.0
    s4 = c1 - camarilla_range * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, EMA, volume, and Camarilla
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(ema_100_12h_aligned[i]) or 
            np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(vol_median_4h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: price vs 12h EMA100
        uptrend = curr_close > ema_100_12h_aligned[i]
        downtrend = curr_close < ema_100_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 4h volume median
        if vol_median_4h[i] <= 0 or np.isnan(vol_median_4h[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_4h[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: price > R4 AND uptrend AND volume spike
            if curr_close > r4_aligned[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price < S4 AND downtrend AND volume spike
            elif curr_close < s4_aligned[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below S4 OR trend turns down
            elif curr_close < s4_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above R4 OR trend turns up
            elif curr_close > r4_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals