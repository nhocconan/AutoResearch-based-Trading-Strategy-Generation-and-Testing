#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Relative Strength Index (RSI) with 1-week trend filter and volume confirmation.
# RSI(14) < 30 = oversold (long opportunity), RSI(14) > 70 = overbought (short opportunity)
# Trend filter: 1-week EMA(34) - only take longs when price > weekly EMA, shorts when price < weekly EMA
# Volume confirmation: volume > 1.5x 20-period average to avoid low-liquidity signals
# Designed for ~15-25 trades/year per symbol with strict entry conditions.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI calculation (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 34-period EMA on 1w close for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # RSI < 30 = oversold (long opportunity)
        # RSI > 70 = overbought (short opportunity)
        if rsi[i] < 30 and close[i] > ema34_1w_aligned[i] and volume_filter[i]:
            # Oversold in uptrend - go long
            signals[i] = 0.25
            position = 1
        elif rsi[i] > 70 and close[i] < ema34_1w_aligned[i] and volume_filter[i]:
            # Overbought in downtrend - go short
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_RSI14_1wEMA34_VolumeFilter"
timeframe = "1d"
leverage = 1.0