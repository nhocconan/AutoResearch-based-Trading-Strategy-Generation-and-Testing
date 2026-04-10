#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and ATR volatility filter
# - Primary signal: Price breaks above/below Camarilla H3/L3 levels derived from 1d OHLC
# - Volume filter: 1d volume > 1.3x 20-period average volume (institutional participation)
# - Volatility filter: 1d ATR(14) < 0.035 * price (low volatility for cleaner breakouts)
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 1.5x ATR(20) on 12h for tighter risk control
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Breakouts capture strong moves; filters avoid chop/false signals
# - Camarilla levels provide mathematically derived support/resistance that works across regimes

name = "12h_1d_camarilla_volume_atr_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla levels (H3, L3, H4, L4)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels based on previous day
    camarilla_high = np.zeros_like(high_1d)
    camarilla_low = np.zeros_like(low_1d)
    camarilla_h3 = np.zeros_like(high_1d)
    camarilla_l3 = np.zeros_like(low_1d)
    camarilla_h4 = np.zeros_like(high_1d)
    camarilla_l4 = np.zeros_like(low_1d)
    
    for i in range(1, len(df_1d)):
        # Previous day's OHLC
        phigh = high_1d[i-1]
        plow = low_1d[i-1]
        pclose = close_1d[i-1]
        
        # Camarilla calculations
        range_val = phigh - plow
        camarilla_high[i] = pclose + range_val * 1.1 / 2
        camarilla_low[i] = pclose - range_val * 1.1 / 2
        camarilla_h3[i] = pclose + range_val * 1.1 / 4
        camarilla_l3[i] = pclose - range_val * 1.1 / 4
        camarilla_h4[i] = pclose + range_val * 1.1 / 2
        camarilla_l4[i] = pclose - range_val * 1.1 / 2
    
    # For first bar, use same values
    camarilla_high[0] = camarilla_high[1] if len(df_1d) > 1 else high_1d[0]
    camarilla_low[0] = camarilla_low[1] if len(df_1d) > 1 else low_1d[0]
    camarilla_h3[0] = camarilla_h3[1] if len(df_1d) > 1 else high_1d[0]
    camarilla_l3[0] = camarilla_l3[1] if len(df_1d) > 1 else low_1d[0]
    camarilla_h4[0] = camarilla_h4[1] if len(df_1d) > 1 else high_1d[0]
    camarilla_l4[0] = camarilla_l4[1] if len(df_1d) > 1 else low_1d[0]
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.3 * avg_volume_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Pre-compute 1d ATR(14) for volatility filter
    tr_1d1 = high_1d - low_1d
    tr_1d2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr_1d3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr_1d1, np.maximum(tr_1d2, tr_1d3))
    tr_1d[0] = tr_1d1[0]
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_filter = (atr_14 / close_1d) < 0.035  # ATR < 3.5% of price
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter)
    
    # Pre-compute 12h ATR(20) for stoploss
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    tr_12h1 = high_12h - low_12h
    tr_12h2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr_12h3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr_12h1, np.maximum(tr_12h2, tr_12h3))
    tr_12h[0] = tr_12h1[0]
    atr_20 = pd.Series(tr_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(atr_filter_aligned[i]) or
            np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price reverts below H3 OR stoploss hit
            if close_12h[i] < camarilla_h3_aligned[i] or close_12h[i] < entry_price - 1.5 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price reverts above L3 OR stoploss hit
            if close_12h[i] > camarilla_l3_aligned[i] or close_12h[i] > entry_price + 1.5 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakouts with volume and volatility filters
            if vol_spike_aligned[i] and atr_filter_aligned[i]:
                # Long: price breaks above H3 level
                if close_12h[i] > camarilla_h3_aligned[i]:
                    position = 1
                    entry_price = close_12h[i]
                    signals[i] = 0.25
                # Short: price breaks below L3 level
                elif close_12h[i] < camarilla_l3_aligned[i]:
                    position = -1
                    entry_price = close_12h[i]
                    signals[i] = -0.25
    
    return signals