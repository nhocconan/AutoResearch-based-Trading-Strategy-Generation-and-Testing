#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R3 AND price > 12h EMA34 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below Camarilla S3 AND price < 12h EMA34 AND volume > 2.0 * avg_volume(20)
# Exit when price crosses Camarilla H4/L4 (mid-point) OR volume < avg_volume(20)
# Uses discrete sizing 0.25 to minimize fee churn
# Target: 50-150 total trades over 4 years (12-37/year) with proper risk control
# Camarilla levels from 12h provide robust support/resistance; 12h EMA34 filters trend; volume spike confirms breakout strength
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)

name = "6h_Camarilla_R3S3_Breakout_12hEMA34_VolumeSpike"
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
    
    # Get 12h data ONCE before loop for Camarilla and EMA34
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough for EMA34 and Camarilla
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate Camarilla levels from previous 12h bar
    # Camarilla: based on previous day's (12h bar's) high, low, close
    # H4 = close + 1.1 * (high - low) / 2
    # L4 = close - 1.1 * (high - low) / 2
    # H3 = close + 1.1 * (high - low) / 4
    # L3 = close - 1.1 * (high - low) / 4
    # H2 = close + 1.1 * (high - low) / 6
    # L2 = close - 1.1 * (high - low) / 6
    # H1 = close + 1.1 * (high - low) / 12
    # L1 = close - 1.1 * (high - low) / 12
    # But we only need R3/S3 (H3/L3) and H4/L4 for exit
    range_12h = high_12h - low_12h
    camarilla_h3 = close_12h + 1.1 * range_12h / 4
    camarilla_l3 = close_12h - 1.1 * range_12h / 4
    camarilla_h4 = close_12h + 1.1 * range_12h / 2
    camarilla_l4 = close_12h - 1.1 * range_12h / 2
    
    # Align Camarilla levels to 6h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Camarilla R3 (H3), above 12h EMA34, volume confirmation
            if close[i] > camarilla_h3_aligned[i] and close[i] > ema34_12h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 (L3), below 12h EMA34, volume confirmation
            elif close[i] < camarilla_l3_aligned[i] and close[i] < ema34_12h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below Camarilla H4/L4 mid-point OR volume drops below average
            camarilla_mid = (camarilla_h4_aligned[i] + camarilla_l4_aligned[i]) / 2
            if close[i] < camarilla_mid or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above Camarilla H4/L4 mid-point OR volume drops below average
            camarilla_mid = (camarilla_h4_aligned[i] + camarilla_l4_aligned[i]) / 2
            if close[i] > camarilla_mid or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals