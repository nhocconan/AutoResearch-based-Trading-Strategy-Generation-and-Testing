#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d volume confirmation and ATR volatility filter
# - Primary signal: Williams %R(14) crosses above -80 for long, below -20 for short on 6h
# - Volume filter: 1d volume > 1.3x 20-period average volume (institutional participation)
# - ATR filter: 1d ATR(14) < 0.05 * price (moderate volatility for cleaner reversals)
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.0x ATR(14) on 6h
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Williams %R captures overbought/oversold reversals; filters avoid chop/false signals

name = "6h_1d_williamsr_volume_atr_v1"
timeframe = "6h"
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
    atr_filter = (atr_14 / close_1d) < 0.05  # ATR < 5% of price
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter)
    
    # Pre-compute 6h Williams %R(14)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_6h) / (highest_high - lowest_low)
    williams_r[highest_high == lowest_low] = -50  # avoid division by zero
    
    # Pre-compute 6h ATR(14) for stoploss
    tr_6h1 = high_6h - low_6h
    tr_6h2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr_6h3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr_6h = np.maximum(tr_6h1, np.maximum(tr_6h2, tr_6h3))
    tr_6h[0] = tr_6h1[0]
    atr_14_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(vol_spike_aligned[i]) or 
            np.isnan(atr_filter_aligned[i]) or np.isnan(atr_14_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses below -50 (mean reversion) OR stoploss hit
            if williams_r[i] < -50 or close_6h[i] < entry_price - 2.0 * atr_14_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 (mean reversion) OR stoploss hit
            if williams_r[i] > -50 or close_6h[i] > entry_price + 2.0 * atr_14_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R reversals with volume and volatility filters
            if vol_spike_aligned[i] and atr_filter_aligned[i]:
                # Long: Williams %R crosses above -80 from below
                if williams_r[i] > -80 and williams_r[i-1] <= -80:
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: Williams %R crosses below -20 from above
                elif williams_r[i] < -20 and williams_r[i-1] >= -20:
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals