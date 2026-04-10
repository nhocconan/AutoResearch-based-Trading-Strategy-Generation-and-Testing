#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and chop regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 1.8x 20-period average AND chop > 61.8 (ranging market)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 1.8x 20-period average AND chop > 61.8 (ranging market)
# - Exit when price returns to Camarilla Pivot level (mean reversion to center)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Camarilla pivots work well in ranging markets which dominate 2025 BTC/ETH action
# - Volume confirmation ensures breakouts have conviction
# - Chop filter avoids trending markets where mean reversion fails

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 12h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Pre-compute 12h Choppiness Index (14-period)
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]  # First bar TR
    for i in range(1, len(high)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(atr_sum / (highest_high - lowest_low)) / log10(14)
    range_hl = highest_high - lowest_low
    chop = np.full_like(close, 50.0, dtype=float)  # Default to neutral
    valid_range = (range_hl > 0) & (~np.isnan(atr_sum))
    chop[valid_range] = 100 * np.log10(atr_sum[valid_range] / range_hl[valid_range]) / np.log10(14)
    
    chop_regime = chop > 61.8  # >61.8 = ranging market (good for mean reversion)
    
    # Pre-compute 1d Camarilla pivot levels (using previous day's OHLC)
    # Camarilla levels: H4, H3, H2, H1, Pivot, L1, L2, L3, L4
    # H3 = Close + 1.1*(High-Low)*1.1/2
    # L3 = Close - 1.1*(High-Low)*1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 2
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 2
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3
    
    # Align HTF indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(vol_ma[i]) or np.isnan(chop_regime[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Camarilla H3 AND volume spike AND chop regime (ranging)
            if (close[i] > camarilla_h3_aligned[i] and 
                volume_spike[i] and 
                chop_regime[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Camarilla L3 AND volume spike AND chop regime (ranging)
            elif (close[i] < camarilla_l3_aligned[i] and 
                  volume_spike[i] and 
                  chop_regime[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to pivot (mean reversion)
            # Exit conditions: price returns to Camarilla Pivot level
            exit_long = (position == 1 and close[i] < camarilla_pivot_aligned[i])
            exit_short = (position == -1 and close[i] > camarilla_pivot_aligned[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals