#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with volume confirmation and 1d choppiness regime filter
# - Long when price breaks above 20-period Donchian upper band AND volume > 1.5x 20-period average AND 1d chop > 61.8 (ranging market)
# - Short when price breaks below 20-period Donchian lower band AND volume > 1.5x 20-period average AND 1d chop > 61.8
# - Exit when price returns to Donchian middle (median of upper/lower) OR chop < 38.2 (trending market)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Donchian breakouts capture sustained moves; chop filter avoids whipsaws in strong trends
# - Volume confirmation reduces false breakouts
# - Works in both bull/bear by trading mean reversion in ranging markets (chop > 61.8)

name = "12h_1d_donchian_chop_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 12h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 12h Donchian channels (20-period)
    donch_hi = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lo = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_hi + donch_lo) / 2.0
    
    # Pre-compute 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) - using Wilder's smoothing
    atr_1d = np.zeros_like(tr)
    if len(tr) >= 14:
        atr_1d[13] = np.mean(tr[1:14])
        for i in range(14, len(tr)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop = np.zeros_like(sum_tr_14)
    mask = (range_14 > 0) & (sum_tr_14 > 0)
    chop[mask] = 100 * np.log10(sum_tr_14[mask] / range_14[mask]) / np.log10(14)
    
    # Chop regime: > 61.8 = ranging (mean revert), < 38.2 = trending
    chop_regime_ranging = chop > 61.8
    
    # Align HTF indicators to 12h timeframe
    chop_regime_ranging_aligned = align_htf_to_ltf(prices, df_1d, chop_regime_ranging)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_hi[i]) or np.isnan(donch_lo[i]) or np.isnan(donch_mid[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop_regime_ranging_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price > Donchian high AND volume spike AND ranging regime
            if (close[i] > donch_hi[i] and 
                volume_spike[i] and 
                chop_regime_ranging_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price < Donchian low AND volume spike AND ranging regime
            elif (close[i] < donch_lo[i] and 
                  volume_spike[i] and 
                  chop_regime_ranging_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to Donchian mid OR chop < 38.2 (trending regime)
            exit_long = (position == 1 and 
                        (close[i] <= donch_mid[i] or not chop_regime_ranging_aligned[i]))
            exit_short = (position == -1 and 
                         (close[i] >= donch_mid[i] or not chop_regime_ranging_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals