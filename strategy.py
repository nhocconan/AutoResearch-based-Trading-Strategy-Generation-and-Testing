#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot (R3/S3) breakout with 1d trend filter (EMA34) and volume confirmation.
# Long when price breaks above 1d Camarilla R3 AND 1d close > 1d EMA34 (uptrend) AND 12h volume > 1.5x 20-period 12h volume MA.
# Short when price breaks below 1d Camarilla S3 AND 1d close < 1d EMA34 (downtrend) AND 12h volume > 1.5x 20-period 12h volume MA.
# Uses 12h timeframe to target 50-150 total trades over 4 years (12-37/year) with tight entry conditions.
# Camarilla levels from 1d provide institutional support/resistance, 1d EMA34 filters trend, volume confirms participation.
# Works in bull/bear markets by only trading breakouts aligned with 1d trend when volume spikes.

name = "12h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels, trend filter, and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels (R3, S3) from previous 1d bar
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2.0
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2.0
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 12h volume 20-period MA for spike detection
    volume_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ma_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: 12h volume > 1.5x 20-period 12h volume MA
        volume_spike = volume[i] > (volume_ma_12h[i] * 1.5)
        
        # Camarilla breakout conditions
        breakout_up = high_val > camarilla_r3_aligned[i]   # Price breaks above 1d Camarilla R3
        breakout_down = low_val < camarilla_s3_aligned[i]  # Price breaks below 1d Camarilla S3
        
        # 1d trend conditions
        trend_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: Camarilla R3 breakout up AND 1d uptrend AND volume spike
            if breakout_up and trend_up and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla S3 breakout down AND 1d downtrend AND volume spike
            elif breakout_down and trend_down and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches 1d Camarilla pivot point (CP) OR trend changes
            # Camarilla pivot point (CP) = (high + low + close) / 3
            camarilla_cp = (high_1d[-1] + low_1d[-1] + close_1d[-1]) / 3.0 if len(high_1d) > 0 else 0
            camarilla_cp_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(close_1d, camarilla_cp))
            if close_val < camarilla_cp_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches 1d Camarilla pivot point (CP) OR trend changes
            if close_val > camarilla_cp_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals