#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily strategy using weekly Bollinger Bands with daily RSI and volume confirmation.
# In weekly uptrend (price above weekly BB middle), buy dips to lower BB with RSI < 40 and volume spike.
# In weekly downtrend (price below weekly BB middle), sell rallies to upper BB with RSI > 60 and volume spike.
# Weekly trend filter ensures alignment with higher timeframe momentum, reducing whipsaw.
# Designed for low trade frequency (10-25/year) to minimize fee drag and capture high-probability mean reversion within trend.

name = "1d_WeeklyBB_RSI_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly Bollinger Bands (20, 2)
    close_series_1w = pd.Series(close_1w)
    bb_middle = close_series_1w.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series_1w.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Align weekly BB to daily timeframe
    bb_middle_aligned = align_htf_to_ltf(prices, df_1w, bb_middle)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1w, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1w, bb_lower)
    
    # Daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 2.0x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for BB
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bb_middle_aligned[i]) or np.isnan(bb_upper_aligned[i]) or
            np.isnan(bb_lower_aligned[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: weekly uptrend (price above weekly BB middle), dip to lower BB with RSI < 40 and volume spike
            if (close[i] > bb_middle_aligned[i] and  # Weekly uptrend
                close[i] <= bb_lower_aligned[i] * 1.01 and  # Near lower BB (allow 1% slack)
                rsi[i] < 40 and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: weekly downtrend (price below weekly BB middle), rally to upper BB with RSI > 60 and volume spike
            elif (close[i] < bb_middle_aligned[i] and  # Weekly downtrend
                  close[i] >= bb_upper_aligned[i] * 0.99 and  # Near upper BB (allow 1% slack)
                  rsi[i] > 60 and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses above weekly BB middle or RSI > 70
            if close[i] >= bb_middle_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses below weekly BB middle or RSI < 30
            if close[i] <= bb_middle_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals