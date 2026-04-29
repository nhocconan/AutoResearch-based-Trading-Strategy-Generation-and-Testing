#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 Breakout with 4h EMA20 Trend Filter and Volume Spike
# Long when price breaks above Camarilla R1 AND price > 4h EMA20 AND volume > 1.5x 24-bar avg
# Short when price breaks below Camarilla S1 AND price < 4h EMA20 AND volume > 1.5x 24-bar avg
# Exit when price retests Camarilla pivot (PP)
# Uses discrete position sizing (0.20) to reduce fee drag. Target: 15-37 trades/year on 1h timeframe.
# Uses 4h trend filter to avoid counter-trend trades, volume confirmation for breakout strength.
# Works in bull via breakout continuation with trend, in bear via breakdown continuation with trend.
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods.

name = "1h_Camarilla_R1S1_Breakout_4hEMA20_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla pivot calculation and EMA20 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate for each 4h bar (using previous bar's data to avoid look-ahead)
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    
    # Set first bar to NaN (no previous bar)
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    prev_close_4h[0] = np.nan
    
    # Calculate Camarilla levels
    camarilla_pp = (prev_high_4h + prev_low_4h + prev_close_4h) / 3.0
    camarilla_range = prev_high_4h - prev_low_4h
    camarilla_r1 = camarilla_pp + camarilla_range * 1.1 / 12.0
    camarilla_s1 = camarilla_pp - camarilla_range * 1.1 / 12.0
    
    # Calculate 4h EMA20 for trend filter
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all 4h indicators to 1h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Volume confirmation: >1.5x 24-bar average volume (6h average)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 1.5 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 20)  # volume MA and EMA20 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(volume_ma_24[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_pp = camarilla_pp_aligned[i]
        curr_r1 = camarilla_r1_aligned[i]
        curr_s1 = camarilla_s1_aligned[i]
        curr_ema20_4h = ema_20_4h_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retests Camarilla pivot (PP)
            if curr_low <= curr_pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price retests Camarilla pivot (PP)
            if curr_high >= curr_pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long when price breaks above R1 AND price > 4h EMA20 AND volume confirmation
            if curr_high > curr_r1 and curr_close > curr_ema20_4h and vol_conf:
                signals[i] = 0.20
                position = 1
            # Short when price breaks below S1 AND price < 4h EMA20 AND volume confirmation
            elif curr_low < curr_s1 and curr_close < curr_ema20_4h and vol_conf:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals