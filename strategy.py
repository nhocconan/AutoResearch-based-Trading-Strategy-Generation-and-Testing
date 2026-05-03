#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d trend filter and volume confirmation.
# Long when price breaks above 1d Camarilla R1 AND 1d close > 1d EMA34 (uptrend) AND 1d volume > 1.3x 20-period volume MA.
# Short when price breaks below 1d Camarilla S1 AND 1d close < 1d EMA34 (downtrend) AND 1d volume > 1.3x 20-period volume MA.
# Uses Camarilla pivot levels from daily timeframe for structure, 1d EMA34 for trend filter, and volume spike for confirmation.
# Designed for 4h timeframe to achieve 75-200 total trades over 4 years (19-50/year) with tight entry conditions.
# Camarilla levels provide institutional reference points, EMA34 filters for trend alignment, volume confirms participation.
# Works in both bull and bear markets by only trading breakouts in the direction of the 1d trend when volume confirms.

name = "4h_Camarilla_R1S1_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (R1, S1)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Calculate 1d EMA34 for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d volume 20-period MA for spike detection
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to LTF (4h)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current 1d volume > 1.3x 20-period MA
        # Use the most recent completed 1d volume (from HTF data)
        vol_1d_idx = len(df_1d) - 1  # Last completed 1d bar
        if vol_1d_idx < 0:
            volume_spike = False
        else:
            current_vol_1d = volume_1d[vol_1d_idx]
            vol_ma_1d_val = volume_ma_1d[vol_1d_idx] if vol_1d_idx < len(volume_ma_1d) else 0
            volume_spike = current_vol_1d > (vol_ma_1d_val * 1.3) if vol_ma_1d_val > 0 else False
        
        # Camarilla breakout conditions
        breakout_up = high_val > camarilla_r1_aligned[i]   # Price breaks above R1
        breakout_down = low_val < camarilla_s1_aligned[i]  # Price breaks below S1
        
        # 1d trend conditions
        trend_up = close_val > ema_34_1d_aligned[i]    # 1d uptrend
        trend_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        
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
            # Camarilla pivot point = (high + low + 2*close)/5 approximated as (R1+S1)/2 for simplicity
            mid_level = (camarilla_r1_aligned[i] + camarilla_s1_aligned[i]) / 2.0
            if close_val < mid_level or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches Camarilla pivot point OR trend changes
            mid_level = (camarilla_r1_aligned[i] + camarilla_s1_aligned[i]) / 2.0
            if close_val > mid_level or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals