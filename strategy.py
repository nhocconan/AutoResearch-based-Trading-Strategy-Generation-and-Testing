#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and ATR filter
# - Primary signal: Price breaks above/below Camarilla H3/L3 levels calculated from 1d OHLC
# - Volume filter: 1d volume > 1.3x 20-period average volume (institutional participation)
# - ATR filter: 1d ATR(14) < 0.035 * price (low volatility for cleaner breakouts)
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 1.5x ATR(20) on 4h
# - Works in bull/bear: Breakouts capture strong moves; filters avoid chop/false signals
# - Target: 20-50 trades/year (80-200 total over 4 years) per 4h strategy guidelines

name = "4h_1d_camarilla_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.3 * avg_volume_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Pre-compute 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr_1d1 = high_1d - low_1d
    tr_1d2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr_1d3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr_1d1, np.maximum(tr_1d2, tr_1d3))
    tr_1d[0] = tr_1d1[0]
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_filter = (atr_14 / close_1d) < 0.035  # ATR < 3.5% of price
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter)
    
    # Pre-compute Camarilla pivot levels from 1d OHLC (H3, L3, H4, L4)
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    #            L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    high_low_range = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * high_low_range * 1.1 / 4
    camarilla_l3 = close_1d - 1.1 * high_low_range * 1.1 / 4
    camarilla_h4 = close_1d + 1.1 * high_low_range * 1.1 / 2
    camarilla_l4 = close_1d - 1.1 * high_low_range * 1.1 / 2
    
    # Al Camarilla levels to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute 4h ATR(20) for stoploss
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    tr_4h1 = high_4h - low_4h
    tr_4h2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr_4h3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    tr_4h[0] = tr_4h1[0]
    atr_20 = pd.Series(tr_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(atr_filter_aligned[i]) or
            np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price retracement to L3 OR stoploss hit
            if close_4h[i] < l3_aligned[i] or close_4h[i] < entry_price - 1.5 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retracement to H3 OR stoploss hit
            if close_4h[i] > h3_aligned[i] or close_4h[i] > entry_price + 1.5 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakouts with volume and volatility filters
            if vol_spike_aligned[i] and atr_filter_aligned[i]:
                # Long: price breaks above H3 (with H4 as stronger confirmation)
                if close_4h[i] > h3_aligned[i]:
                    position = 1
                    entry_price = close_4h[i]
                    signals[i] = 0.25
                # Short: price breaks below L3 (with L4 as stronger confirmation)
                elif close_4h[i] < l3_aligned[i]:
                    position = -1
                    entry_price = close_4h[i]
                    signals[i] = -0.25
    
    return signals