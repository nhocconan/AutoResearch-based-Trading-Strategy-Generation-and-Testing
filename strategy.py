#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Breakout with 1w volume regime filter
# - Primary: 12h timeframe for lower frequency and reduced fee drag
# - HTF: 1w for volume regime (avoid low-volume false breakouts)
# - Long: Price breaks above H3 Camarilla pivot (1d) + 1w volume > 1.5x 4-week MA
# - Short: Price breaks below L3 Camarilla pivot (1d) + 1w volume > 1.5x 4-week MA
# - Exit: Price reverts to Camarilla Pivot Point (mean reversion)
# - Position sizing: 0.25 (discrete level)
# - Target: 50-150 total trades over 4 years (12-37/year) - within 12h sweet spot
# - Works in bull/bear: Camarilla pivots capture mean reversion in ranging markets (2025) and breakouts in trending markets
# - Volume regime filter reduces false breakouts in low-volume environments

name = "12h_1w_camarilla_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute 12h OHLCV
    open_12h = prices['open'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pre-compute 1w data
    volume_1w = df_1w['volume'].values
    
    # Calculate 12h Camarilla Pivot Points (based on previous 1d)
    # Align daily OHLC to 12h bars (using previous day's OHLC)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate Camarilla levels for each 12h bar (using previous day's OHLC)
    rng = high_1d_aligned - low_1d_aligned
    h3 = close_1d_aligned + 1.25 * rng  # Long entry: break above H3
    l3 = close_1d_aligned - 1.25 * rng  # Short entry: break below L3
    pivot = (high_1d_aligned + low_1d_aligned + close_1d_aligned) / 3.0  # Mean reversion exit
    
    # Calculate 1w volume moving average (4-period) for volume regime filter
    volume_ma_4_1w = pd.Series(volume_1w).rolling(window=4, min_periods=4).mean().values
    volume_ma_4_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_4_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(volume_ma_4_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume regime condition: current 1w volume > 1.5x 4-week MA
        volume_regime = volume_1w[i] > 1.5 * volume_ma_4_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above H3 resistance + volume regime
            if (close_12h[i] > h3[i] and volume_regime):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below L3 support + volume regime
            elif (close_12h[i] < l3[i] and volume_regime):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit condition: Price reverts to Pivot Point (mean reversion)
            
            if position == 1:  # Long position
                exit_condition = close_12h[i] < pivot[i]  # Reverted to pivot
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = close_12h[i] > pivot[i]  # Reverted to pivot
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals