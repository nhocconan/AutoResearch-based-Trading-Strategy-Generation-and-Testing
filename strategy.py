#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels with volume confirmation and ATR regime filter
# Camarilla levels (H3/L3 for fade, H4/L4 for breakout) work well in both trending and ranging markets
# Volume confirmation ensures breakouts have conviction
# ATR regime filter (current ATR > 20-period mean) avoids low-volatility chop
# Position sizing: fixed 0.25 (25% of capital) for consistent risk management
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1d_camarilla_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    h4 = close_1d + range_1d * 1.1 / 2
    h3 = close_1d + range_1d * 1.1 / 4
    l3 = close_1d - range_1d * 1.1 / 4
    l4 = close_1d - range_1d * 1.1 / 2
    
    # Calculate 1d ATR (14-period) for regime filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all levels to 6h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ATR regime filter (current ATR > 20-period mean)
    atr_ma_20 = pd.Series(atr_aligned).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(h4_aligned[i]) or np.isnan(h3_aligned[i]) or
            np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(atr_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(atr_ma_20[i]) or atr_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.3x average 6h volume
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        # ATR regime filter: only trade when volatility is elevated (avoid chop)
        vol_regime = atr_aligned[i] > atr_ma_20[i]
        
        if not (volume_confirmed and vol_regime):
            signals[i] = 0.0
            continue
        
        # Fixed position size for consistent risk management
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit on retracement to H3 level (fade level)
            if close[i] < h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to L3 level (fade level)
            if close[i] > l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Camarilla trading logic:
            # Fade at H3/L3 (mean reversion in range)
            # Breakout continuation at H4/L4 (trend following)
            if close[i] > h4_aligned[i] and volume_confirmed:
                # Breakout above H4 - go long
                position = 1
                signals[i] = position_size
            elif close[i] < l4_aligned[i] and volume_confirmed:
                # Breakdown below L4 - go short
                position = -1
                signals[i] = -position_size
            elif close[i] < h3_aligned[i] and close[i] > l3_aligned[i]:
                # Inside H3-L3 range - look for mean reversion
                # Long near L3, short near H3
                if close[i] <= (l3_aligned[i] + h3_aligned[i]) / 2:
                    # Lower half - bias long near L3
                    if close[i] < l3_aligned[i] * 1.005:  # Near L3
                        position = 1
                        signals[i] = position_size
                else:
                    # Upper half - bias short near H3
                    if close[i] > h3_aligned[i] * 0.995:  # Near H3
                        position = -1
                        signals[i] = -position_size
    
    return signals