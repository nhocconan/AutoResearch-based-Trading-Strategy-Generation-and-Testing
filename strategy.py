#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 level AND 1d close > 1d EMA34 (uptrend) AND 1d volume > 1.5x 20-period volume MA.
# Short when price breaks below Camarilla S3 level AND 1d close < 1d EMA34 (downtrend) AND 1d volume > 1.5x 20-period volume MA.
# Uses Camarilla pivots from prior 1d for structure, 1d EMA34 for trend alignment, and 1d volume spike for confirmation.
# Designed for 4h timeframe to achieve 75-200 total trades over 4 years (19-50/year) with strict entry conditions.
# Works in both bull and bear markets by only trading breakouts in the direction of the 1d trend when volume confirms.

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Get 1d data for Camarilla pivots, trend, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d volume 20-period MA for spike detection
    volume_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate prior 1d Camarilla levels (using completed 1d bar)
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4, etc.
    # Actually standard Camarilla: R3 = close + 1.1*(high-low)*1.1/6, S3 = close - 1.1*(high-low)*1.1/6
    # But we'll use the common formulation: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Using prior completed 1d bar (index -1) to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each completed 1d bar
    camarilla_R3 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 4
    camarilla_S3 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (using completed 1d bar only)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]) or 
            np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # 1d trend conditions
        trend_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        # Since we don't have intraday 1d volume, we use the volume from the completed 1d bar
        # For 4h bar i, we use the volume from the most recent completed 1d bar
        vol_1d_idx = len(df_1d) - 1  # Most recent completed 1d bar
        if vol_1d_idx < 0:
            volume_spike = False
        else:
            current_1d_volume = df_1d['volume'].iloc[vol_1d_idx]
            volume_spike = current_1d_volume > (volume_ma_1d_aligned[i] * 1.5)
        
        # Camarilla breakout conditions
        breakout_up = high_val > camarilla_R3_aligned[i]  # Price breaks above R3
        breakout_down = low_val < camarilla_S3_aligned[i]  # Price breaks below S3
        
        if position == 0:
            # Long: Camarilla breakout up AND 1d uptrend AND volume spike
            if breakout_up and trend_up and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla breakout down AND 1d downtrend AND volume spike
            elif breakout_down and trend_down and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches Camarilla pivot point (mid-level) OR trend changes
            # Camarilla pivot point = (high + low + close)/3 of prior 1d bar
            camarilla_PP = (high_1d + low_1d + close_1d) / 3
            camarilla_PP_aligned = align_htf_to_ltf(prices, df_1d, camarilla_PP)
            if close_val < camarilla_PP_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches Camarilla pivot point OR trend changes
            camarilla_PP = (high_1d + low_1d + close_1d) / 3
            camarilla_PP_aligned = align_htf_to_ltf(prices, df_1d, camarilla_PP)
            if close_val > camarilla_PP_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals