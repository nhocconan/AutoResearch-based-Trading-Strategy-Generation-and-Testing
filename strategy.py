#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams Alligator (jaw/teeth/lips) with 6h Donchian(20) breakout and volume confirmation.
# Williams Alligator defines market regime: when lines are intertwined (chop) vs aligned (trend).
# In choppy regimes (Alligator sleeping): fade Donchian breakouts at 20-period levels.
# In trending regimes (Alligator awake): continue Donchian breakouts in direction of trend.
# Uses 1d Alligator for higher-timeframe regime filter to reduce false breakouts on 6h.
# Position size 0.25 balances return and drawdown. Discrete levels minimize fee churn.
# Target: 50-150 total trades over 4 years = 12-37/year for 6h (within proven winning range).

name = "6h_WilliamsAlligator_Donchian20_Regime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator regime filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate 1d Williams Alligator: SMA(close, 13), SMA(close, 8), SMA(close, 5) with offsets
    close_1d = df_1d['close'].values
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().shift(8).values  # Blue line
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().shift(5).values   # Red line
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(3).values    # Green line
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Alligator regime: TRUE when intertwined (chop/sleeping), FALSE when aligned (trend/awake)
    # Chop condition: max - min of the three lines < 0.5% of price (tightly coiled)
    max_line = np.maximum(np.maximum(jaw_aligned, teeth_aligned), lips_aligned)
    min_line = np.minimum(np.minimum(jaw_aligned, teeth_aligned), lips_aligned)
    alligator_range = max_line - min_line
    price_level = (jaw_aligned + teeth_aligned + lips_aligned) / 3
    chop_threshold = 0.005 * price_level  # 0.5% of price
    is_chop = alligator_range < chop_threshold  # TRUE = chop/sleeping, FALSE = trend/awake
    
    # Calculate 6h Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume confirmation: >1.8x 30-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_30 = volume_series.rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > 1.8 * volume_ma_30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_ma_30[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high[i] and volume_spike[i]
        short_breakout = close[i] < donchian_low[i] and volume_spike[i]
        
        # Regime-based logic
        if is_chop[i]:  # Alligator sleeping -> chop regime -> fade breakouts
            # In chop: fade Donchian breakouts (expect reversion to mean)
            long_entry = short_breakout  # Short on upside breakout
            short_entry = long_breakout  # Long on downside breakout
            # Exit when price returns to Alligator midline (teeth)
            long_exit = close[i] > teeth_aligned[i]
            short_exit = close[i] < teeth_aligned[i]
        else:  # Alligator awake -> trend regime -> continue breakouts
            # In trend: continue Donchian breakouts in direction of Alligator alignment
            # Determine trend direction from Alligator alignment (lips > teeth > jaw = uptrend)
            is_uptrend = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
            is_downtrend = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
            long_entry = long_breakout and is_uptrend
            short_entry = short_breakout and is_downtrend
            # Exit when trend weakens (Alligator lines start to intertwine) or opposite Donchian breakout
            long_exit = is_chop[i] or short_breakout
            short_exit = is_chop[i] or long_breakout
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals