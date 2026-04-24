#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h for balanced trade frequency and signal quality.
- HTF: 12h EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Volume: Current 4h volume > 2.0 * 24-period volume MA to capture institutional interest.
- Entry: Long when close breaks above Camarilla R3 AND 12h EMA34 bullish AND volume spike.
         Short when close breaks below Camarilla S3 AND 12h EMA34 bearish AND volume spike.
- Exit: Opposite Camarilla level (S3 for long, R3 for short) or loss of volume confirmation.
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 100-180 total trades over 4 years (25-45/year) for 4h timeframe.
This strategy uses Camarilla pivot levels as dynamic support/resistance, which work well in both
trending and ranging markets. The 12h EMA34 filter ensures we trade with the higher timeframe trend,
while volume spikes confirm institutional participation. Designed to avoid overtrading (<400 total 4h trades).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels (based on previous day's OHLC)
    # We need daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    # Camarilla: R4 = close + ((high-low) * 1.1/2), R3 = close + ((high-low) * 1.1/4)
    #          S3 = close - ((high-low) * 1.1/4), S4 = close - ((high-low) * 1.1/2)
    # We use R3 and S3 as our breakout levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid division by zero and handle first bar
    range_hl = prev_high - prev_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)  # Prevent zero range
    
    camarilla_r3 = prev_close + (range_hl * 1.1 / 4)
    camarilla_s3 = prev_close - (range_hl * 1.1 / 4)
    
    # Align daily Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34
    df_12h_close = df_12h['close'].values
    ema_12h = pd.Series(df_12h_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 24-period 12h volume MA
    df_12h_volume = df_12h['volume'].values
    vol_ma_12h = pd.Series(df_12h_volume).rolling(window=24, min_periods=24).mean().values
    
    # Align HTF indicators to 4h
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Volume confirmation: current 4h volume > 2.0 * 24-period 12h volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_12h_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 24)  # Need enough bars for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        camarilla_r3 = camarilla_r3_aligned[i]
        camarilla_s3 = camarilla_s3_aligned[i]
        ema_val = ema_12h_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: Break above Camarilla R3 AND 12h EMA34 bullish (close > EMA)
                if curr_high > camarilla_r3 and curr_close > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Break below Camarilla S3 AND 12h EMA34 bearish (close < EMA)
                elif curr_low < camarilla_s3 and curr_close < ema_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Break below Camarilla S3 OR loss of volume confirmation
            if curr_low < camarilla_s3 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Break above Camarilla R3 OR loss of volume confirmation
            if curr_high > camarilla_r3 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0