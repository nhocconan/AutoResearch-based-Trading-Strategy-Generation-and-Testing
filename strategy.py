#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume regime filter
# - Long: Price breaks above Camarilla H3 level (1d) + 1d volume > 1.5x 20-period MA (increased participation)
# - Short: Price breaks below Camarilla L3 level (1d) + 1d volume > 1.5x 20-period MA
# - Exit: Price returns to Camarilla Pivot point (mean reversion) OR volume regime ends
# - Position sizing: 0.25 discrete level
# - Works in bull/bear: Camarilla levels act as support/resistance in ranging markets and breakout points in trending markets.
#   Volume filter ensures breakouts have conviction. Targets ~25-50 trades/year.

name = "4h_1d_camarilla_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1d Camarilla levels (based on previous day)
    # Camarilla: H4, H3, H2, H1, L1, L2, L3, L4
    # Pivot = (high + low + close) / 3
    # Range = high - low
    # H3 = pivot + (range * 1.1/4)
    # L3 = pivot - (range * 1.1/4)
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    camarilla_h3_1d = pivot_1d + (range_1d * 1.1 / 4.0)
    camarilla_l3_1d = pivot_1d - (range_1d * 1.1 / 4.0)
    camarilla_pivot_1d = pivot_1d
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot_1d)
    
    # Calculate 1d volume regime: current volume > 1.5x 20-period MA
    volume_ma_20_1d = pd.Series(volume_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    volume_regime = align_htf_to_ltf(prices, df_1d, volume_1d) > 1.5 * volume_ma_20_1d_aligned
    
    # Calculate 4h volume confirmation: current volume > 20-period MA
    volume_ma_20_4h = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(volume_regime[i]) or 
            np.isnan(volume_ma_20_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 20-period MA
        vol_confirm_4h = volume[i] > volume_ma_20_4h[i]
        
        if position == 0:  # Flat - look for Camarilla breakouts
            # Long entry: Price breaks above Camarilla H3 + 4h vol confirmation + 1d volume regime
            if close[i] > camarilla_h3_aligned[i] and vol_confirm_4h and volume_regime[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Camarilla L3 + 4h vol confirmation + 1d volume regime
            elif close[i] < camarilla_l3_aligned[i] and vol_confirm_4h and volume_regime[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price returns to Camarilla Pivot OR 1d volume regime ends
            if position == 1:  # Long position
                if close[i] <= camarilla_pivot_aligned[i] or not volume_regime[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] >= camarilla_pivot_aligned[i] or not volume_regime[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals