#!/usr/bin/env python3
"""
Hypothesis: A 1-day strategy using weekly Bollinger Band breakouts with volume confirmation and 
weekly trend filter. The strategy targets major trend continuations while avoiding whipsaws 
in choppy markets. Weekly Bollinger Bands provide dynamic support/resistance, and volume 
confirms institutional participation. Designed for lower trade frequency (target: 10-25 trades/year)
to minimize fee drag on higher timeframe.
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
    
    # Load weekly data for Bollinger Bands and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly Bollinger Bands (20-period, 2 std dev)
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    
    # Weekly trend filter: 50-period EMA
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: weekly volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume_1w > (1.5 * vol_ma_20)
    
    # Align weekly indicators to daily timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    vol_confirm_aligned = align_htf_to_ltf(prices, df_1w, vol_confirm.astype(float))
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if weekly data not ready
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_confirm_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper BB + above weekly EMA50 + volume confirmation
            if (close[i] > upper_band_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                vol_confirm_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower BB + below weekly EMA50 + volume confirmation
            elif (close[i] < lower_band_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  vol_confirm_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite band or trend changes
            if position == 1:
                if (close[i] < lower_band_aligned[i] or 
                    close[i] < ema_50_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > upper_band_aligned[i] or 
                    close[i] > ema_50_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_weekly_bb_breakout_volume_trend"
timeframe = "1d"
leverage = 1.0