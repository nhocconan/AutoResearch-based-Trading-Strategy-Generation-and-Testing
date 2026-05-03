#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 level AND 1w close > 1w EMA34 (uptrend) AND 1d volume > 1.5x 20-period volume MA.
# Short when price breaks below Camarilla S3 level AND 1w close < 1w EMA34 (downtrend) AND 1d volume > 1.5x 20-period volume MA.
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for 1d timeframe to achieve 30-100 total trades over 4 years (7-25/year).
# Camarilla levels provide mathematically derived support/resistance, 1w EMA34 filters for primary trend alignment, volume confirms institutional participation.
# Works in both bull and bear markets by only trading breakouts in the direction of the 1w trend when volume confirms.

name = "1d_Camarilla_R3S3_1wEMA34_VolumeSpike"
timeframe = "1d"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend direction
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d Camarilla levels (based on previous day's range)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use previous day's high/low to calculate today's levels (no look-ahead)
    prev_high = pd.Series(high).shift(1).values
    prev_low = pd.Series(low).shift(1).values
    prev_close = pd.Series(close).shift(1).values
    
    # Calculate Camarilla R3 and S3 levels
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume 20-period MA for spike detection
    volume_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        volume_val = volume[i]
        
        # Volume spike condition: current volume > 1.5x 20-period MA
        volume_spike = volume_val > (volume_ma_1d_aligned[i] * 1.5)
        
        # 1w trend conditions
        trend_up = close_val > ema_34_1w_aligned[i]   # 1w uptrend
        trend_down = close_val < ema_34_1w_aligned[i]  # 1w downtrend
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND 1w uptrend AND volume spike
            if close_val > camarilla_r3[i] and trend_up and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 AND 1w downtrend AND volume spike
            elif close_val < camarilla_s3[i] and trend_down and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches Camarilla Pivot Point (mid-level) OR trend changes
            camarilla_pp = (prev_high[i] + prev_low[i] + prev_close[i]) / 3.0  # Classic pivot point
            if close_val < camarilla_pp or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches Camarilla Pivot Point OR trend changes
            camarilla_pp = (prev_high[i] + prev_low[i] + prev_close[i]) / 3.0  # Classic pivot point
            if close_val > camarilla_pp or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals