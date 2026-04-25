#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeFilter
Hypothesis: Daily Donchian(20) breakout with 1-week EMA50 trend filter and volume confirmation.
In uptrend (close > 1w EMA50): long upper band breakout. In downtrend: short lower band breakout.
Volume spike (>2.0x 20-day average) confirms breakout strength. Exit on opposite band touch.
Designed for low trade frequency (~15/year) and robustness in bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA50 for trend direction
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian channels (20-day) on daily data
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(high_ma_20[i]) or 
            np.isnan(low_ma_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1w_aligned[i]
        upper_band = high_ma_20[i]
        lower_band = low_ma_20[i]
        
        if position == 0:
            # Regime-based entry logic
            if close[i] > ema_trend:  # Uptrend regime
                # Long: break above upper band with volume confirmation
                long_signal = (close[i] > upper_band) and vol_spike[i]
            else:  # Downtrend regime
                # Short: break below lower band with volume confirmation
                short_signal = (close[i] < lower_band) and vol_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: touch or cross below lower band
            if close[i] < lower_band:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: touch or cross above upper band
            if close[i] > upper_band:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0