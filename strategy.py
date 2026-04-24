#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h for optimal trade frequency and lower fee drag.
- HTF: 1d EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Volume: Current 4h volume > 2.0 * 20-period 4h volume MA to capture institutional interest.
- Entry: Long when price breaks above R3 AND 1d EMA34 bullish AND volume spike.
         Short when price breaks below S3 AND 1d EMA34 bearish AND volume spike.
- Exit: Opposite Camarilla level (S3 for long, R3 for short) or loss of volume confirmation.
- Signal size: 0.30 discrete to balance return and drawdown.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
This strategy leverages Camarilla pivot points as dynamic support/resistance levels,
with trend filtering to avoid counter-trend trades and volume confirmation to ensure
institutional participation. Works in both bull and bear markets by only taking trades
in the direction of the 1d trend.
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
    
    # Calculate 4h Camarilla levels from previous day
    # Need daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels: based on previous day's high, low, close
    # R4 = close + ((high - low) * 1.5/2)
    # R3 = close + ((high - low) * 1.25/2)
    # R2 = close + ((high - low) * 1.1/2)
    # R1 = close + ((high - low) * 1.05/2)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 1.05/2)
    # S2 = close - ((high - low) * 1.1/2)
    # S3 = close - ((high - low) * 1.25/2)
    # S4 = close - ((high - low) * 1.5/2)
    
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    rang = prev_high - prev_low
    R3 = prev_close + (rang * 1.25 / 2)
    S3 = prev_close - (rang * 1.25 / 2)
    
    # Align Camarilla levels to 4h
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    
    # Get 4h data for EMA50 trend filter and volume MA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend filter
    df_4h_close = df_4h['close'].values
    ema_4h = pd.Series(df_4h_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period 4h volume MA
    df_4h_volume = df_4h['volume'].values
    vol_ma_4h = pd.Series(df_4h_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h (though already 4h, alignment ensures proper timing)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Volume confirmation: current 4h volume > 2.0 * 20-period 4h volume MA
    volume_spike = volume > (2.0 * vol_ma_4h_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need enough bars for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(R3_4h[i]) or np.isnan(S3_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_val = ema_4h_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: price breaks above R3 AND 4h EMA34 bullish (close > EMA)
                if curr_high > R3_4h[i] and curr_close > ema_val:
                    signals[i] = 0.30
                    position = 1
                # Bearish: price breaks below S3 AND 4h EMA34 bearish (close < EMA)
                elif curr_low < S3_4h[i] and curr_close < ema_val:
                    signals[i] = -0.30
                    position = -1
        elif position == 1:
            # Long exit: price breaks below S3 OR loss of volume confirmation
            if curr_low < S3_4h[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price breaks above R3 OR loss of volume confirmation
            if curr_high > R3_4h[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0