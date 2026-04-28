#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Uses Williams Alligator (Jaw/Teeth/Lips) from 4h data for trend identification
# Only takes trades when price is outside the Alligator's mouth AND aligned with 1d EMA34 trend
# Volume confirmation (>1.5x average) filters weak breakouts
# Designed to work in both bull and bear markets by using Alligator's convergence/divergence
# Target: 20-50 trades/year via tight Alligator conditions + volume + trend filter

name = "4h_WilliamsAlligator_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe (completed 1d candles only)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Williams Alligator on 4h data (Smoothed Median Price)
    # Median Price = (High + Low) / 2
    median_price = (high + low) / 2
    
    # Smoothed Median Price (SMA of median price)
    smoothed_median = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Alligator lines: Jaw (13-period, 8-bar shift), Teeth (8-period, 5-bar shift), Lips (5-period, 3-bar shift)
    jaw = pd.Series(smoothed_median).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(smoothed_median).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(smoothed_median).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13, 8, 5) + 8  # Need sufficient history for Alligator and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_confirm[i]
        price = close[i]
        ema34_val = ema34_1d_aligned[i]
        
        # Alligator conditions
        # Mouth open (trending): Lips > Teeth > Jaw (uptrend) or Lips < Teeth < Jaw (downtrend)
        # Mouth closed (ranging): lines intertwined
        lips_val = lips[i]
        teeth_val = teeth[i]
        jaw_val = jaw[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price above Lips AND Lips > Teeth > Jaw (uptrend) AND price > 1d EMA34 AND volume confirm
            if price > lips_val and lips_val > teeth_val and teeth_val > jaw_val and price > ema34_val and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price below Lips AND Lips < Teeth < Jaw (downtrend) AND price < 1d EMA34 AND volume confirm
            elif price < lips_val and lips_val < teeth_val and teeth_val < jaw_val and price < ema34_val and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when Alligator closes or reverses
            # Exit on Alligator closing (Lips < Teeth) or price crosses below Teeth
            if lips_val < teeth_val or price < teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when Alligator closes or reverses
            # Exit on Alligator closing (Lips > Teeth) or price crosses above Teeth
            if lips_val > teeth_val or price > teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals