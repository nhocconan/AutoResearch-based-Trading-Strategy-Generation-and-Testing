#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla H4/L4 breakout with 1w EMA50 trend filter and volume spike confirmation.
# Long when price breaks above 4h Camarilla H4 level AND 1w close > 1w EMA50 (uptrend) AND 4h volume > 2.0x 20-period volume MA.
# Short when price breaks below 4h Camarilla L4 level AND 1w close < 1w EMA50 (downtrend) AND 4h volume > 2.0x 20-period volume MA.
# Exit on retracement to 4h Camarilla H3/L3 levels or trend reversal.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size 0.25.
# Designed for 4h timeframe to achieve 75-200 total trades over 4 years (19-50/year) with strict entry conditions.
# Camarilla H4/L4 levels are stronger breakout levels than R1/S1, reducing false breakouts. 1w EMA50 filters for higher-timeframe trend alignment.
# Volume spike confirmation ensures institutional participation. Works in both bull and bear markets by only trading breakouts in the direction of the 1w trend when volume confirms.

name = "4h_Camarilla_H4L4_Breakout_1wEMA50_VolumeSpike_Session"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend direction
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Camarilla levels (previous week's OHLC for H4/L4)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels using previous 1d OHLC (shifted by 1 to avoid look-ahead)
    df_1d_shifted = df_1d.copy()
    df_1d_shifted['open'] = df_1d_shifted['open'].shift(1)
    df_1d_shifted['high'] = df_1d_shifted['high'].shift(1)
    df_1d_shifted['low'] = df_1d_shifted['low'].shift(1)
    df_1d_shifted['close'] = df_1d_shifted['close'].shift(1)
    
    # Calculate Camarilla levels using previous day's OHLC
    prev_high = df_1d_shifted['high'].values
    prev_low = df_1d_shifted['low'].values
    prev_close = df_1d_shifted['close'].values
    
    # Calculate the range
    range_hl = prev_high - prev_low
    
    # Calculate Camarilla levels (H4 and L4 are stronger breakout levels)
    camarilla_h4 = prev_close + 1.1 * range_hl    # H4 level for breakout
    camarilla_l4 = prev_close - 1.1 * range_hl    # L4 level for breakout
    camarilla_h3 = prev_close + 1.1 * range_hl    # H3 level for exit (same as H4 in standard Camarilla)
    camarilla_l3 = prev_close - 1.1 * range_hl    # L3 level for exit (same as L4 in standard Camarilla)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d_shifted, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d_shifted, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d_shifted, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d_shifted, camarilla_l3)
    
    # Calculate 4h volume 20-period MA for spike detection
    volume_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_ma_4h[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current 4h volume > 2.0x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_4h[i] * 2.0)
        
        # Camarilla H4/L4 breakout conditions
        breakout_up = high_val > camarilla_h4_aligned[i]  # Price breaks above H4 level
        breakout_down = low_val < camarilla_l4_aligned[i]  # Price breaks below L4 level
        
        # 1w trend conditions
        trend_up = close_val > ema_50_1w_aligned[i]   # 1w uptrend
        trend_down = close_val < ema_50_1w_aligned[i]  # 1w downtrend
        
        if position == 0:
            # Long: Camarilla H4 breakout up AND 1w uptrend AND volume spike AND session
            if breakout_up and trend_up and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla L4 breakout down AND 1w downtrend AND volume spike AND session
            elif breakout_down and trend_down and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches Camarilla H3/L3 levels OR trend changes
            if close_val < camarilla_h3_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches Camarilla H3/L3 levels OR trend changes
            if close_val > camarilla_l3_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals