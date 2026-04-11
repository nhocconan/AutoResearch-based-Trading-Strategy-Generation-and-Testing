#!/usr/bin/env python3
# 12h_1w_donchian_breakout_volume_v1
# Strategy: 12h Donchian breakout with weekly volume confirmation and trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Donchian breakouts on 12h timeframe capture medium-term momentum. Weekly volume surge confirms institutional interest. Trend filter (price vs weekly EMA50) avoids counter-trend trades. Designed for low trade frequency (12-37/year) to minimize fee drag and work in both bull and bear markets by trading breakouts in direction of weekly trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_donchian_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Donchian channels on 12h (20-period)
    period = 20
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    upper_band = highest_high.values
    lower_band = lowest_low.values
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Weekly volume average (20-period) for confirmation
    volume_1w = df_1w['volume'].values
    vol_avg_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_20_1w)
    
    # Align raw weekly volume for current bar comparison
    vol_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or \
           np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg_20_1w_aligned[i]) or \
           np.isnan(vol_1w_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current weekly volume > 20-period average
        vol_confirm = vol_1w_aligned[i] > vol_avg_20_1w_aligned[i]
        
        # Trend filter: close vs weekly EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Entry conditions
        # Long: Price breaks above upper band AND uptrend AND volume confirmation
        if not np.isnan(upper_band[i]) and close[i] > upper_band[i] and uptrend and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price breaks below lower band AND downtrend AND volume confirmation
        elif not np.isnan(lower_band[i]) and close[i] < lower_band[i] and downtrend and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price crosses opposite band
        elif position == 1 and close[i] < lower_band[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > upper_band[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals