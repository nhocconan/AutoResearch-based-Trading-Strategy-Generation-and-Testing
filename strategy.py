#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot + 1d volume spike + chop regime filter
# - Primary signal: 12h price touching Camarilla H3 (short) or L3 (long) levels from 1d
# - Volume confirmation: 1d volume > 1.5x 20-period average volume (avoid low-participation)
# - Regime filter: 1d choppiness index > 61.8 (range market) for mean reversion at pivots
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Camarilla pivots work in ranges, volume spike confirms participation, chop filter avoids trends

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for today using yesterday's OHLC
    # H3 = Close + 1.1*(High-Low)/2, L3 = Close - 1.1*(High-Low)/2
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Pre-compute 1d volume regime: volume > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_regime = volume_1d > (1.5 * avg_volume_20)
    volume_regime_aligned = align_htf_to_ltf(prices, df_1d, volume_regime)
    
    # Pre-compute 1d choppiness index: CHOP > 61.8 = range (mean revert)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(range(14))) / log10(14)
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(abs(high_1d - close_1d.shift(1)))
    tr3 = pd.Series(abs(low_1d - close_1d.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).sum().values
    high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = high_14 - low_14
    chop = np.where(range_14 != 0,
                    100 * np.log10(atr_14 / range_14) / np.log10(14),
                    50)  # neutral when no range
    chop_regime = chop > 61.8  # range market
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_regime_aligned[i]) or
            np.isnan(chop_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_price = prices['close'].iloc[i]
        
        if position == 1:  # Long position
            # Exit: price crosses above H3 (take profit) or below L3 (stop)
            if close_price > camarilla_h3_aligned[i] or close_price < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses below L3 (take profit) or above H3 (stop)
            if close_price < camarilla_l3_aligned[i] or close_price > camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla level touches with volume and chop confirmation
            # Long: price touches L3 from above AND volume regime AND chop regime (range)
            if (abs(close_price - camarilla_l3_aligned[i]) < 0.001 * close_price and  # touch L3
                volume_regime_aligned[i] and 
                chop_regime_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price touches H3 from below AND volume regime AND chop regime (range)
            elif (abs(close_price - camarilla_h3_aligned[i]) < 0.001 * close_price and  # touch H3
                  volume_regime_aligned[i] and 
                  chop_regime_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals