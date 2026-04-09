#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian breakout with volume confirmation and chop filter
# Donchian(20) from 12h provides structure aligned with 4h timeframe
# Volume confirmation (current 4h volume > 1.5x 20-period average) filters false breakouts
# Choppiness regime filter: CHOP(14) > 61.8 for mean reversion, < 38.2 for trend following
# Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# Works in bull/bear: price reacts to 12h structure, volume confirms validity, chop filter adapts to regime
# Discrete position sizing: 0.0, ±0.25 to minimize fee churn

name = "4h_12h_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    donchian_h_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_l_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_h_20 + donchian_l_20) / 2.0
    
    # Align Donchian levels to 4h timeframe
    dh_20_aligned = align_htf_to_ltf(prices, df_12h, donchian_h_20)
    dl_20_aligned = align_htf_to_ltf(prices, df_12h, donchian_l_20)
    dm_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid)
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute Choppiness Index (14-period) for regime filter
    # CHOP = 100 * LOG10(SUM(ATR(1),14) / (LOG10(HH(14)-LL(14)) / LOG10(14)))
    # Simplified: CHOP = 100 * LOG10(ATR_sum / (LOG10(range_max - range_min) / LOG10(period)))
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.maximum(tr1, np.absolute(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]  # First bar TR
    atr_1 = pd.Series(tr1).rolling(window=1, min_periods=1).sum().values  # ATR(1) = true range
    atr_sum_14 = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = hh_14 - ll_14
    # Avoid division by zero and log of zero
    chop_raw = 100 * np.log10(atr_sum_14 / (np.log10(np.maximum(range_14, 1e-10)) / np.log10(14)))
    chop = np.where(range_14 > 0, chop_raw, 50.0)  # Default to 50 when no range
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(dh_20_aligned[i]) or np.isnan(dl_20_aligned[i]) or
            np.isnan(dm_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 4h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit conditions: Donchian middle retracement OR chop > 61.8 (range) AND price < mid
            if close[i] < dm_aligned[i] or (chop[i] > 61.8 and close[i] < dm_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: Donchian middle retracement OR chop > 61.8 (range) AND price > mid
            if close[i] > dm_aligned[i] or (chop[i] > 61.8 and close[i] > dm_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout trading with volume confirmation and chop regime filter
            # Long on Donchian high breakout in trending OR mean reversion from low in ranging
            # Short on Donchian low breakdown in trending OR mean reversion from high in ranging
            if volume_confirmed:
                if chop[i] < 38.2:  # Strong trend - follow breakout
                    if close[i] > dh_20_aligned[i]:
                        position = 1
                        signals[i] = 0.25
                    elif close[i] < dl_20_aligned[i]:
                        position = -1
                        signals[i] = -0.25
                elif chop[i] > 61.8:  # Range - mean reversion
                    if close[i] < dl_20_aligned[i]:  # Oversold - buy
                        position = 1
                        signals[i] = 0.25
                    elif close[i] > dh_20_aligned[i]:  # Overbought - sell
                        position = -1
                        signals[i] = -0.25
                # Neutral chop (38.2-61.8): no new entries
    
    return signals