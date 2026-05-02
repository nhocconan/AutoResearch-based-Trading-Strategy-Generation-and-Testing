#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA21 trend filter and volume spike confirmation
# Williams Alligator identifies trend via three smoothed SMAs (Jaw, Teeth, Lips)
# Price above all three lines = uptrend; below all three = downtrend
# 1d EMA21 filters for higher-timeframe trend alignment to avoid counter-trend trades
# Volume spike (>2.0 x 30-period EMA) confirms breakout validity and reduces false signals
# Discrete position sizing (0.25) balances opportunity with fee drag control
# Target: 50-150 total trades over 4 years (12-37/year) for optimal risk-adjusted returns

name = "12h_WilliamsAlligator_1dEMA21_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation (volume spike > 2.0 x 30-period EMA)
    vol_ema_30 = pd.Series(volume).ewm(span=30, adjust=False, min_periods=30).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_30)
    
    # 1d data for EMA21 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA21 for trend filter
    ema_21_1d = pd.Series(df_1d['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Williams Alligator on 12h timeframe (primary)
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars
    # Lips: 5-period SMMA, shifted 3 bars
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    jaw = pd.Series(close).ewm(alpha=1/13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(alpha=1/8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(alpha=1/5, adjust=False, min_periods=5).mean().values
    
    # Shift the lines as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Set NaN for shifted values that roll in from the end
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(ema_21_1d_aligned[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from Williams Alligator
        # Uptrend: Lips > Teeth > Jaw and price > Lips
        # Downtrend: Lips < Teeth < Jaw and price < Lips
        uptrend = (lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i]) and (close[i] > lips_shifted[i])
        downtrend = (lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i]) and (close[i] < lips_shifted[i])
        
        if position == 0:  # Flat - look for new entries
            # Long: Uptrend from Alligator + price above Lips + volume confirmation + 1d EMA21 uptrend
            if uptrend and volume_confirmation[i] and close[i] > ema_21_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Downtrend from Alligator + price below Lips + volume confirmation + 1d EMA21 downtrend
            elif downtrend and volume_confirmation[i] and close[i] < ema_21_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Downtrend from Alligator OR price closes below Teeth OR 1d trend changes to downtrend
            if downtrend or close[i] < teeth_shifted[i] or close[i] < ema_21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Uptrend from Alligator OR price closes above Teeth OR 1d trend changes to uptrend
            if uptrend or close[i] > teeth_shifted[i] or close[i] > ema_21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals