#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume spike and chop regime filter
# - Long when price breaks above Donchian(20) high with 1d volume > 2.0x 20-period average and CHOP(14) < 38.2 (trending)
# - Short when price breaks below Donchian(20) low with 1d volume > 2.0x 20-period average and CHOP(14) < 38.2
# - Uses 4h timeframe targeting 20-50 trades/year (80-200 total over 4 years) to minimize fee drag
# - 1d volume confirmation ensures institutional participation
# - CHOP filter avoids ranging markets where breakouts fail
# - ATR-based stoploss: exit when price moves against position by 2.5x ATR(14)
# - Discrete position sizing (0.25) to minimize fee churn

name = "4h_1d_donchian_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d volume confirmation: > 2.0x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 1d ATR(14) for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1d = np.zeros_like(tr)
    atr_14_1d[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14_1d[i] = (atr_14_1d[i-1] * (14-1) + tr[i]) / 14
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # 4h Choppiness Index (CHOP) - regime filter
    # CHOP < 38.2 = trending (good for breakouts), CHOP > 61.8 = ranging (avoid breakouts)
    tr_4h = np.maximum(high_4h - low_4h,
                       np.maximum(np.abs(high_4h - np.roll(close_4h, 1)),
                                  np.abs(low_4h - np.roll(close_4h, 1))))
    tr_4h[0] = high_4h[0] - low_4h[0]
    atr_14_4h = np.zeros_like(tr_4h)
    atr_14_4h[14-1] = np.mean(tr_4h[:14])
    for i in range(14, len(tr_4h)):
        atr_14_4h[i] = (atr_14_4h[i-1] * (14-1) + tr_4h[i]) / 14
    
    # Sum of true range over 14 periods
    sum_tr_14 = pd.Series(tr_4h).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index: CHOP = 100 * log10(sum_tr_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    hh_ll_diff = hh_14 - ll_14
    chop_raw = np.zeros_like(sum_tr_14)
    mask = (hh_ll_diff > 0) & ~np.isnan(sum_tr_14) & ~np.isnan(hh_ll_diff)
    chop_raw[mask] = 100 * np.log10(sum_tr_14[mask] / hh_ll_diff[mask]) / np.log10(14)
    
    # CHOP < 38.2 indicates trending market (favorable for breakouts)
    chop_filter = chop_raw < 38.2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(vol_spike_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or price breaks below Donchian low
            if (prices['close'].iloc[i] < entry_price - 2.5 * entry_atr or 
                prices['close'].iloc[i] < donchian_low[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or price breaks above Donchian high
            if (prices['close'].iloc[i] > entry_price + 2.5 * entry_atr or 
                prices['close'].iloc[i] > donchian_high[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume and chop filters
            if vol_spike_1d_aligned[i] and chop_filter[i]:
                # Long signal: price breaks above Donchian high
                if prices['close'].iloc[i] > donchian_high[i]:
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_1d_aligned[i]
                    signals[i] = 0.25
                # Short signal: price breaks below Donchian low
                elif prices['close'].iloc[i] < donchian_low[i]:
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_1d_aligned[i]
                    signals[i] = -0.25
    
    return signals