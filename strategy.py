#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla pivot breakouts with 1d trend filter and volume confirmation.
# Enter long when price breaks above Camarilla R3 level and 1d close > 1d EMA34 and volume > 2x 20-bar average.
# Enter short when price breaks below Camarilla S3 level and 1d close < 1d EMA34 and volume > 2x 20-bar average.
# Exit when price reaches opposite Camarilla level (R1 for longs, S1 for shorts) or trend weakens.
# Uses discrete position sizing (0.25) to manage risk and minimize fee churn.
# Camarilla levels provide precise intraday support/resistance, EMA34 filters trend direction, volume confirms conviction.
# Works in bull markets (breakout continuation) and bear markets (mean reversion at extreme levels).

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter (MTF structure)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 12h data for Camarilla pivots (MTF structure)
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Typical price for Camarilla calculation
    typical_price = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    # Camarilla levels (based on previous 12h bar)
    # R4 = close + range * 1.1/2
    # R3 = close + range * 1.1/4
    # R2 = close + range * 1.1/6
    # R1 = close + range * 1.1/12
    # S1 = close - range * 1.1/12
    # S2 = close - range * 1.1/6
    # S3 = close - range * 1.1/4
    # S4 = close - range * 1.1/2
    camarilla_r3 = close_12h + range_12h * 1.1 / 4.0
    camarilla_s3 = close_12h - range_12h * 1.1 / 4.0
    camarilla_r1 = close_12h + range_12h * 1.1 / 12.0
    camarilla_s1 = close_12h - range_12h * 1.1 / 12.0
    
    # Align 12h Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    
    # Volume confirmation: >2x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # Ensure sufficient history for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Trend filter: 1d close vs EMA34
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        # Camarilla breakout conditions
        breakout_up = close[i] > r3_aligned[i-1]  # Break above previous period's R3
        breakout_down = close[i] < s3_aligned[i-1]  # Break below previous period's S3
        
        # Exit conditions
        exit_long = close[i] < r1_aligned[i] or (not uptrend and position == 1)
        exit_short = close[i] > s1_aligned[i] or (not downtrend and position == -1)
        
        # Handle entries and exits
        if breakout_up and uptrend and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
        elif breakout_down and downtrend and vol_confirm and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals