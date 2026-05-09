# 1d KAMA + RSI + Chop Filter Strategy
# Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
# RSI(14) for momentum confirmation, and Choppiness Index to avoid ranging markets.
# Enter long when KAMA turns up, RSI > 50, and market is trending (CHOP < 38.2).
# Enter short when KAMA turns down, RSI < 50, and market is trending (CHOP < 38.2).
# Exit when opposite KAMA signal occurs or RSI reaches extreme levels.
# Designed to capture sustained trends while avoiding whipsaws in choppy markets.
# Weekly trend filter ensures alignment with higher timeframe momentum.
# Target: 20-40 trades over 4 years (5-10/year) with size 0.25.

name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    def kama(close, er_len=10, fast_len=2, slow_len=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else np.zeros_like(change)
        # Avoid division by zero
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_len+1) - 2/(slow_len+1)) + 2/(slow_len+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Calculate Choppiness Index
    def choppiness_index(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr1 = np.abs(high - low)
        tr2 = np.abs(np.roll(high, 1) - close)
        tr3 = np.abs(np.roll(low, 1) - close)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        # First TR is just high-low
        tr[0] = tr1[0]
        # Calculate ATR using Wilder's smoothing
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Calculate max/min range over period
        max_h = np.zeros_like(close)
        min_l = np.zeros_like(close)
        for i in range(len(close)):
            if i < period:
                max_h[i] = np.max(high[:i+1])
                min_l[i] = np.min(low[:i+1])
            else:
                max_h[i] = np.max(high[i-period+1:i+1])
                min_l[i] = np.min(low[i-period+1:i+1])
        
        # Avoid division by zero
        range_hl = max_h - min_l
        choppiness = np.where(range_hl != 0, 
                              100 * np.log10(np.sum(atr[i-period+1:i+1]) / range_hl) / np.log10(period), 
                              50)
        return choppiness
    
    # Calculate RSI
    def rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate daily indicators
    kama_val = kama(close, er_len=10, fast_len=2, slow_len=30)
    chop_val = choppiness_index(high, low, close, period=14)
    rsi_val = rsi(close, period=14)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicator calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_val[i]) or np.isnan(chop_val[i]) or np.isnan(rsi_val[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA turning up, RSI > 50, trending market (CHOP < 38.2), weekly uptrend
            if (kama_val[i] > kama_val[i-1] and 
                rsi_val[i] > 50 and 
                chop_val[i] < 38.2 and 
                ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA turning down, RSI < 50, trending market (CHOP < 38.2), weekly downtrend
            elif (kama_val[i] < kama_val[i-1] and 
                  rsi_val[i] < 50 and 
                  chop_val[i] < 38.2 and 
                  ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA turning down OR RSI overbought (>70) OR choppy market
            if (kama_val[i] < kama_val[i-1] or 
                rsi_val[i] > 70 or 
                chop_val[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA turning up OR RSI oversold (<30) OR choppy market
            if (kama_val[i] > kama_val[i-1] or 
                rsi_val[i] < 30 or 
                chop_val[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals