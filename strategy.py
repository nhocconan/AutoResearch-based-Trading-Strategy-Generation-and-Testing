#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d/21 EMA trend filter and volume confirmation
# Williams Alligator uses three smoothed moving averages (Jaw, Teeth, Lips) to identify trends
# Jaw (13-period): Blue line, Teeth (8-period): Red line, Lips (5-period): Green line
# Trend is bullish when Lips > Teeth > Jaw, bearish when Lips < Teeth < Jaw
# 1d EMA21 provides higher-timeframe trend alignment to avoid counter-trend trades
# Volume spike (>1.5 x 20-period EMA) confirms breakout validity
# Works in bull markets (bullish alignment + 1d EMA21 up) and bear markets (bearish alignment + 1d EMA21 down)
# Uses discrete position sizing (0.30) to minimize fee churn and control drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag

name = "12h_WilliamsAlligator_1dEMA21_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h Williams Alligator: three smoothed moving averages
    # Jaw (13-period, smoothed by 8 periods)
    sma_13 = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(sma_13).rolling(window=8, min_periods=8).mean().values
    
    # Teeth (8-period, smoothed by 5 periods)
    sma_8 = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(sma_8).rolling(window=5, min_periods=5).mean().values
    
    # Lips (5-period, smoothed by 3 periods)
    sma_5 = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(sma_5).rolling(window=3, min_periods=3).mean().values
    
    # 1d data for trend filter (EMA21)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # 1d EMA21 calculation
    ema_21_1d = pd.Series(df_1d['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Volume confirmation (volume spike > 1.5 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Alligator calculation)
    start_idx = 21
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_21_1d_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Determine trend bias from 1d EMA21
        uptrend = close[i] > ema_21_1d_aligned[i]
        downtrend = close[i] < ema_21_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish Alligator alignment with volume confirmation and uptrend
            if bullish_alignment and volume_confirmation[i] and uptrend:
                signals[i] = 0.30
                position = 1
            # Short: Bearish Alligator alignment with volume confirmation and downtrend
            elif bearish_alignment and volume_confirmation[i] and downtrend:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Loss of bullish alignment OR trend changes to downtrend
            if not bullish_alignment or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: Loss of bearish alignment OR trend changes to uptrend
            if not bearish_alignment or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals