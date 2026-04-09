#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h volume spike and 1d choppiness regime filter
# - Primary signal: 4h price breaks above/below 20-period Donchian channel
# - Volume confirmation: 12h volume > 1.5x 20-period median volume (avoid low-participation breakouts)
# - Regime filter: 1d choppiness index > 61.8 (range market) for mean reversion, < 38.2 (trend) for trend continuation
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Stoploss: ATR-based exit when price moves 2.5x ATR against position
# - Works in bull/bear: Donchian captures breakouts in trends, choppiness filter avoids false signals in ranges
# - Target: 20-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines

name = "4h_12h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h volume regime
    vol_12h = df_12h['volume'].values
    median_vol_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).median().values
    vol_regime_12h = align_htf_to_ltf(prices, df_12h, vol_12h > (1.5 * median_vol_12h))
    
    # Pre-compute 1d choppiness index
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])],
                        np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Max HH-LL over 14 periods
    max_hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    max_hh_ll_14 = max_hh - min_ll
    
    # Choppiness Index = 100 * log10(sum_tr_14 / max_hh_ll_14) / log10(14)
    chop_raw = np.where(max_hh_ll_14 > 0,
                        100 * np.log10(sum_tr_14 / max_hh_ll_14) / np.log10(14),
                        50)  # neutral when no range
    chop_1d = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # Pre-compute 4h Donchian channels
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Pre-compute 4h ATR for stoploss
    tr_4h1 = np.abs(high_4h[1:] - low_4h[:-1])
    tr_4h2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr_4h3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.concatenate([[np.max([high_4h[0] - low_4h[0], np.abs(high_4h[0] - close_4h[0]), np.abs(low_4h[0] - close_4h[0])])],
                           np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))])
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_regime_12h[i]) or np.isnan(chop_1d[i]) or
            np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian middle OR stoploss hit
            if close_4h[i] < donchian_middle[i] or close_4h[i] < entry_price - 2.5 * atr_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian middle OR stoploss hit
            if close_4h[i] > donchian_middle[i] or close_4h[i] > entry_price + 2.5 * atr_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and chop regime filter
            # Long: price breaks above Donchian upper AND volume regime AND chop < 38.2 (trending)
            if (close_4h[i] > donchian_upper[i] and 
                vol_regime_12h[i] and 
                chop_1d[i] < 38.2):
                position = 1
                entry_price = close_4h[i]
                signals[i] = 0.25
            # Short: price breaks below Donchian lower AND volume regime AND chop < 38.2 (trending)
            elif (close_4h[i] < donchian_lower[i] and 
                  vol_regime_12h[i] and 
                  chop_1d[i] < 38.2):
                position = -1
                entry_price = close_4h[i]
                signals[i] = -0.25
            # Mean reversion in ranging markets: chop > 61.8
            # Long: price touches Donchian lower AND volume regime AND chop > 61.8 (range)
            elif (close_4h[i] <= donchian_lower[i] * 1.001 and  # slight tolerance for touch
                  vol_regime_12h[i] and 
                  chop_1d[i] > 61.8):
                position = 1
                entry_price = close_4h[i]
                signals[i] = 0.25
            # Short: price touches Donchian upper AND volume regime AND chop > 61.8 (range)
            elif (close_4h[i] >= donchian_upper[i] * 0.999 and  # slight tolerance for touch
                  vol_regime_12h[i] and 
                  chop_1d[i] > 61.8):
                position = -1
                entry_price = close_4h[i]
                signals[i] = -0.25
    
    return signals