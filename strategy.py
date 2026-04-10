#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + chop regime filter
# - Primary signal: Price breaks above/below 4h Donchian(20) channel
# - Trend filter: 1d volume > 1.5x 20-period average (institutional participation)
# - Regime filter: 4h Chopiness Index(14) > 61.8 (range) for mean reversion, < 38.2 (trend) for trend follow
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.0x ATR(14) on 4h
# - Target: 19-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Donchian breakouts capture trends; chop filter adapts to ranging markets

name = "4h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.5 * avg_volume_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Pre-compute 4h Donchian(20) channels
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    highest_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h ATR(14) for stoploss
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 4h Chopiness Index(14)
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop[sum_tr == 0] = 50  # Avoid division by zero/log of zero
    chop[np.isnan(chop)] = 50
    chop[np.isinf(chop)] = 50
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(atr_14[i]) or np.isnan(chop[i]) or
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price crosses below Donchian lower OR stoploss hit
            if close_4h[i] < lowest_low[i] or close_4h[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above Donchian upper OR stoploss hit
            if close_4h[i] > highest_high[i] or close_4h[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume and chop filters
            if vol_spike_aligned[i]:
                # Long: Price breaks above Donchian upper in trending market (chop < 38.2)
                if close_4h[i] > highest_high[i] and chop[i] < 38.2:
                    position = 1
                    entry_price = close_4h[i]
                    signals[i] = 0.25
                # Short: Price breaks below Donchian lower in trending market (chop < 38.2)
                elif close_4h[i] < lowest_low[i] and chop[i] < 38.2:
                    position = -1
                    entry_price = close_4h[i]
                    signals[i] = -0.25
                # Mean reversion in ranging market (chop > 61.8)
                elif chop[i] > 61.8:
                    # Long: Price breaks below Donchian lower (oversold bounce)
                    if close_4h[i] < lowest_low[i]:
                        position = 1
                        entry_price = close_4h[i]
                        signals[i] = 0.25
                    # Short: Price breaks above Donchian upper (overbought rejection)
                    elif close_4h[i] > highest_high[i]:
                        position = -1
                        entry_price = close_4h[i]
                        signals[i] = -0.25
    
    return signals