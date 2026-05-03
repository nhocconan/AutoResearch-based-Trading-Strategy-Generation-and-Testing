#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator (Jaw=13, Teeth=8, Lips=5) with 1w EMA34 trend filter and volume confirmation
# Williams Alligator identifies trendless markets when lines are intertwined (choppy) and trending when aligned.
# In choppy markets (Alligator sleeping), we fade extremes at 1w Donchian channels.
# In trending markets (Alligator awakened), we follow the 1w EMA34 direction on breakouts.
# Volume confirmation (1.5x 20-period EMA) filters low-momentum false signals.
# Designed for 30-100 total trades over 4 years (7-25/year) with discrete sizing to minimize fee drag.
# Works in both bull and bear markets by adapting to regime (choppy vs trending).

name = "1d_WilliamsAlligator_1wEMA34_VolumeSpike_Regime"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Williams Alligator: Smoothed Median Price (typical price) with different periods
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    tp_values = typical_price.values
    
    # Jaw: 13-period SMMA (Smoothed Moving Average) of median price, shifted 8 bars
    jaw = pd.Series(tp_values).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    teeth = pd.Series(tp_values).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMMA of median price, shifted 3 bars
    lips = pd.Series(tp_values).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 1d timeframe (already aligned, but using helper for consistency)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get 1w data for EMA34 trend filter and Donchian channels (for regime-based exits)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1w Donchian channels (20-period) for breakout signals in trending regime
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_upper_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().shift(1).values
    donchian_upper_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_1w)
    donchian_lower_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_1w)
    
    # Volume confirmation: 20-period EMA on 1d volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    regime = 0    # 0: undefined, 1: trending (Alligator awake), -1: choppy (Alligator sleeping)
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(donchian_upper_1w_aligned[i]) or 
            np.isnan(donchian_lower_1w_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine regime: Alligator awakening (trending) vs sleeping (choppy)
        # Trending: Lips > Teeth > Jaw (uptrend) OR Lips < Teeth < Jaw (downtrend)
        # Choppy: lines are intertwined (not clearly separated)
        lips_above_teeth = lips_aligned[i] > teeth_aligned[i]
        teeth_above_jaw = teeth_aligned[i] > jaw_aligned[i]
        lips_below_teeth = lips_aligned[i] < teeth_aligned[i]
        teeth_below_jaw = teeth_aligned[i] < jaw_aligned[i]
        
        is_uptrend = lips_above_teeth and teeth_above_jaw
        is_downtrend = lips_below_teeth and teeth_below_jaw
        is_trending = is_uptrend or is_downtrend
        
        if is_trending:
            regime = 1 if is_uptrend else -1  # 1 for uptrend, -1 for downtrend
        else:
            regime = 0  # choppy/ranging
        
        # Volume spike: current volume > 1.5 x 20-period EMA
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            if regime != 0 and volume_spike:  # Trending regime with volume confirmation
                # Follow Alligator direction (trend following)
                if regime == 1:  # Uptrend
                    if close[i] > donchian_upper_1w_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                else:  # Downtrend (regime == -1)
                    if close[i] < donchian_lower_1w_aligned[i]:
                        signals[i] = -0.25
                        position = -1
            elif regime == 0 and volume_spike:  # Choppy regime with volume confirmation
                # Mean reversion: fade extremes at 1w Donchian channels
                if close[i] > donchian_upper_1w_aligned[i]:
                    signals[i] = -0.25  # Short at upper band
                    position = -1
                elif close[i] < donchian_lower_1w_aligned[i]:
                    signals[i] = 0.25   # Long at lower band
                    position = 1
        elif position == 1:  # Long position
            # Exit conditions
            if regime == 1:  # In uptrend: exit on trend reversal or Donchian lower break
                if regime != 1 or close[i] < donchian_lower_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # In choppy or downtrend regime: exit long on mean reversion or trend change
                if regime == -1 or close[i] < lips_aligned[i]:  # Exit on trend change or mean reversion to lips
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit conditions
            if regime == -1:  # In downtrend: exit on trend reversal or Donchian upper break
                if regime != -1 or close[i] > donchian_upper_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # In choppy or uptrend regime: exit short on mean reversion or trend change
                if regime == 1 or close[i] > lips_aligned[i]:  # Exit on trend change or mean reversion to lips
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals