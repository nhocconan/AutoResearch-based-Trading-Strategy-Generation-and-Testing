#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and ATR filter
# - Primary signal: Price breaks above/below Camarilla H3/L3 levels calculated from prior 1d candle
# - Volume filter: 1d volume > 1.5x 20-period average volume (ensures strong participation)
# - ATR filter: 1d ATR(14) < 0.04 * price (moderate volatility for sustainable breakouts)
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 1.5x ATR(14) on 12h
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines

name = "12h_1d_camarilla_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla pivot levels (H3, L3) from prior day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and ranges
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    
    # Camarilla levels: H3 = close + range * 1.1/4, L3 = close - range * 1.1/4
    camarilla_h3 = close_1d + (range_hl * 1.1 / 4.0)
    camarilla_l3 = close_1d - (range_hl * 1.1 / 4.0)
    
    # Align Camarilla levels to 12h timeframe (use prior day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.5 * avg_volume_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Pre-compute 1d ATR(14) for volatility filter
    tr_1d1 = high_1d - low_1d
    tr_1d2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr_1d3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr_1d1, np.maximum(tr_1d2, tr_1d3))
    tr_1d[0] = tr_1d1[0]
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_filter = (atr_14 / close_1d) < 0.04  # ATR < 4% of price
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter)
    
    # Pre-compute 12h ATR(14) for stoploss
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    tr_12h1 = high_12h - low_12h
    tr_12h2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr_12h3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr_12h1, np.maximum(tr_12h2, tr_12h3))
    tr_12h[0] = tr_12h1[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(atr_filter_aligned[i]) or
            np.isnan(atr_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reversion to Camarilla H3 level OR stoploss hit
            if close_12h[i] < camarilla_h3_aligned[i] or close_12h[i] < entry_price - 1.5 * atr_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reversion to Camarilla L3 level OR stoploss hit
            if close_12h[i] > camarilla_l3_aligned[i] or close_12h[i] > entry_price + 1.5 * atr_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakouts with volume and volatility filters
            if vol_spike_aligned[i] and atr_filter_aligned[i]:
                # Long: price breaks above Camarilla H3
                if close_12h[i] > camarilla_h3_aligned[i]:
                    position = 1
                    entry_price = close_12h[i]
                    signals[i] = 0.25
                # Short: price breaks below Camarilla L3
                elif close_12h[i] < camarilla_l3_aligned[i]:
                    position = -1
                    entry_price = close_12h[i]
                    signals[i] = -0.25
    
    return signals