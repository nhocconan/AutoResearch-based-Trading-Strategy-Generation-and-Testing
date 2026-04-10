#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with 1d volume confirmation and chop regime filter
# - Primary signal: Williams %R(14) crosses below -80 (oversold) for long, above -20 (overbought) for short
# - Volume filter: 1d volume > 1.3x 50-period average volume (institutional participation)
# - Chop filter: 1d Choppiness Index(14) > 61.8 (ranging market favors mean reversion)
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.5x ATR(14) on 12h
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Mean reversion in ranging markets; volume confirms institutional interest

name = "12h_1d_williamsr_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    avg_volume_50 = pd.Series(volume_1d).rolling(window=50, min_periods=50).mean().values
    vol_spike = volume_1d > (1.3 * avg_volume_50)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Pre-compute 1d Choppiness Index(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index = 100 * log10(sumTR/(HH-LL)) / log10(14)
    # Avoid division by zero
    hl_range = hh - ll
    chop_raw = np.where(hl_range > 0, tr_sum / hl_range, 1.0)
    chop = 100 * np.log10(chop_raw) / np.log10(14)
    chop_filter = chop > 61.8  # Chop > 61.8 = ranging market
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter)
    
    # Pre-compute 12h Williams %R(14)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Highest high and lowest low over 14 periods
    hh_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (HH - Close) / (HH - LL) * -100
    hl_range_12h = hh_12h - ll_12h
    williams_r = np.where(hl_range_12h > 0, ((hh_12h - close_12h) / hl_range_12h) * -100, -50.0)
    
    # Pre-compute 12h ATR(14) for stoploss
    tr_12h1 = high_12h - low_12h
    tr_12h2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr_12h3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr_12h1, np.maximum(tr_12h2, tr_12h3))
    tr_12h[0] = tr_12h1[0]
    atr_14 = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(vol_spike_aligned[i]) or 
            np.isnan(chop_filter_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses above -50 (mean reversion) OR stoploss hit
            if williams_r[i] > -50.0 or close_12h[i] < entry_price - 2.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -50 (mean reversion) OR stoploss hit
            if williams_r[i] < -50.0 or close_12h[i] > entry_price + 2.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extremes with volume and chop filters
            if vol_spike_aligned[i] and chop_filter_aligned[i]:
                # Long: Williams %R crosses below -80 (oversold)
                if williams_r[i] < -80.0 and williams_r[max(0, i-1)] >= -80.0:
                    position = 1
                    entry_price = close_12h[i]
                    signals[i] = 0.25
                # Short: Williams %R crosses above -20 (overbought)
                elif williams_r[i] > -20.0 and williams_r[max(0, i-1)] <= -20.0:
                    position = -1
                    entry_price = close_12h[i]
                    signals[i] = -0.25
    
    return signals