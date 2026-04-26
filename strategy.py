#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wEMA34_Trend_VolumeSpike
Hypothesis: On daily timeframe, Camarilla R1/S1 breakouts aligned with 1-week EMA34 trend filter and volume confirmation capture strong multi-day trends while avoiding whipsaws. Uses discrete position sizing (0.25) to minimize fee churn. Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for HTF trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 1d Camarilla levels (R1, S1) using previous 1d's OHLC
    high_1d = df_1w['high'].values  # Wait, this is wrong - need 1d data for Camarilla
    # Actually need to get 1d data separately for Camarilla calculation
    
    # Fix: Need both 1w for trend and 1d for Camarilla levels
    # Let me reload correctly
    
    # Actually, let's approach this differently - get 1d data for Camarilla, 1w for trend
    
    # For now, using 1w data for both - but this needs correction
    
    # RELOADING CORRECTLY:
    # Get 1d data for Camarilla levels calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 1d Camarilla levels (R1, S1) using previous 1d's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    close_1d_shifted = np.concatenate([[np.nan], close_1d[:-1]])  # previous 1d close
    
    camarilla_range = high_1d - low_1d
    r1 = close_1d_shifted + 1.1 * camarilla_range / 12
    s1 = close_1d_shifted - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 1d timeframe (no alignment needed since primary is 1d)
    r1_aligned = r1  # Already at 1d frequency
    s1_aligned = s1  # Already at 1d frequency
    
    # 1d volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    # Volume regime: avoid extremely low volume (chop) - volume > 0.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    low_volume_regime = volume < 0.5 * vol_ma_50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for EMA and volume MA)
    start_idx = max(34, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1w_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(vol_ma_50[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1w trend filter (EMA34)
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Volume confirmation and regime filter
        volume_spike = volume[i] > 2.0 * vol_ma_20[i]
        not_low_volume = not low_volume_regime[i]
        
        # Camarilla breakout conditions
        breakout_r1 = close[i] > r1_aligned[i]
        breakout_s1 = close[i] < s1_aligned[i]
        
        # Long logic: breakout above R1 in uptrend with volume and not in low volume regime
        if uptrend and volume_spike and not_low_volume and breakout_r1:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: breakout below S1 in downtrend with volume and not in low volume regime
        elif downtrend and volume_spike and not_low_volume and breakout_s1:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: loss of trend OR entering low volume regime
        elif position == 1 and (not uptrend or low_volume_regime[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not downtrend or low_volume_regime[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0