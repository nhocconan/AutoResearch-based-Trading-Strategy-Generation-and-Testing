#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h 12h Camarilla pivot with volume confirmation and 1d trend filter.
# Fade at R3/S3 levels in ranging markets, breakout continuation at R4/S4 in trending markets.
# Uses Camarilla pivot levels from 12h timeframe, volume confirmation, and 1d EMA(50) trend filter.
# Designed for 12-30 trades/year on 6h timeframe with focus on mean reversion in ranges and breakout continuation in trends.

name = "6h_12h_camarilla_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels (using previous 12h bar)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla levels for each 12h bar (based on previous bar's range)
    camarilla_r4 = np.full_like(high_12h, np.nan)
    camarilla_r3 = np.full_like(high_12h, np.nan)
    camarilla_s3 = np.full_like(low_12h, np.nan)
    camarilla_s4 = np.full_like(low_12h, np.nan)
    
    for i in range(1, len(df_12h)):
        # Previous bar's close, high, low
        prev_close = close_12h[i-1]
        prev_high = high_12h[i-1]
        prev_low = low_12h[i-1]
        range_ = prev_high - prev_low
        
        # Camarilla calculations
        camarilla_r4[i] = prev_close + range_ * 1.1 / 2
        camarilla_r3[i] = prev_close + range_ * 1.1 / 4
        camarilla_s3[i] = prev_close - range_ * 1.1 / 4
        camarilla_s4[i] = prev_close - range_ * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume moving average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after volume MA period
        # Skip if any required data is invalid
        if (np.isnan(r4_12h_aligned[i]) or np.isnan(r3_12h_aligned[i]) or 
            np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.2 * 20-period average volume
        vol_filter = volume[i] > 1.2 * vol_ma_20[i]
        
        # Determine 1d trend direction
        is_uptrend = close[i] > ema_50_1d_aligned[i]
        is_downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Mean reversion at S3/R3 (fade) - only in ranging markets (when price near EMA)
        near_ema = abs(close[i] - ema_50_1d_aligned[i]) < (ema_50_1d_aligned[i] * 0.02)  # Within 2% of EMA
        long_fade = (close[i] <= s3_12h_aligned[i]) and vol_filter and near_ema
        short_fade = (close[i] >= r3_12h_aligned[i]) and vol_filter and near_ema
        
        # Breakout continuation at S4/R4 - only in trending markets
        long_breakout = (close[i] >= s4_12h_aligned[i]) and vol_filter and is_uptrend
        short_breakout = (close[i] <= r4_12h_aligned[i]) and vol_filter and is_downtrend
        
        # Entry conditions
        long_entry = long_fade or long_breakout
        short_entry = short_fade or short_breakout
        
        # Exit conditions: opposite signal or volatility exhaustion
        exit_long = short_entry
        exit_short = long_entry
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals