#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + chop regime filter
# - Long when price breaks above Donchian(20) high with 1d volume > 2.0x 20-period average and chop < 61.8 (trending)
# - Short when price breaks below Donchian(20) low with 1d volume > 2.0x 20-period average and chop < 61.8 (trending)
# - Uses 4h timeframe targeting 20-50 trades/year (80-200 total over 4 years) to minimize fee drag
# - 1d volume confirmation ensures breakout strength
# - Chop regime filter (CHOP < 61.8) ensures we only trade in trending markets, avoiding whipsaws in ranging markets
# - ATR-based stoploss: exit when price moves against position by 2.5x ATR(14) or Donchian breakout in opposite direction
# - Discrete position sizing (0.25) to minimize fee churn

name = "4h_1d_donchian_volume_chop_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ATR(14) for chop calculation
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
    
    # 1d True Range sum for chop calculation (sum of TR over 14 periods)
    tr_sum_14 = np.zeros_like(tr)
    tr_sum_14[14-1] = np.sum(tr[:14])
    for i in range(14, len(tr)):
        tr_sum_14[i] = tr_sum_14[i-1] - tr[i-14] + tr[i]
    tr_sum_14_aligned = align_htf_to_ltf(prices, df_1d, tr_sum_14)
    
    # 1d ATR(14) * 100 for chop denominator
    atr_100_1d = atr_14_1d * 100
    atr_100_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_100_1d)
    
    # 1d Chop calculation: CHOP = 100 * log10(TR_sum_14 / (ATR_14 * 14)) / log10(14)
    # Avoid division by zero and log of zero
    with np.errstate(divide='ignore', invalid='ignore'):
        chop_1d = 100 * np.log10(tr_sum_14_aligned / (atr_100_1d_aligned * 14 + 1e-10)) / np.log10(14)
        chop_1d = np.where((tr_sum_14_aligned > 0) & (atr_100_1d_aligned > 0), chop_1d, 50.0)  # default to 50 when invalid
    
    # 1d volume confirmation: > 2.0x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Donchian high: highest high over 20 periods
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian low: lowest low over 20 periods
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # 4h ATR(14) for stoploss
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr_14_4h = np.zeros_like(tr_4h)
    atr_14_4h[14-1] = np.mean(tr_4h[:14])
    for i in range(14, len(tr_4h)):
        atr_14_4h[i] = (atr_14_4h[i-1] * (14-1) + tr_4h[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(chop_1d[i]) or 
            np.isnan(atr_14_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or Donchian breakout below low (opposite signal)
            if (prices['close'].iloc[i] < entry_price - 2.5 * entry_atr or 
                prices['close'].iloc[i] < donchian_low[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or Donchian breakout above high (opposite signal)
            if (prices['close'].iloc[i] > entry_price + 2.5 * entry_atr or 
                prices['close'].iloc[i] > donchian_high[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume and chop filters
            if vol_spike_1d_aligned[i] and chop_1d[i] < 61.8:  # Trending regime
                # Long signal: price breaks above Donchian high
                if prices['close'].iloc[i] > donchian_high[i]:
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_4h[i]
                    signals[i] = 0.25
                # Short signal: price breaks below Donchian low
                elif prices['close'].iloc[i] < donchian_low[i]:
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_4h[i]
                    signals[i] = -0.25
    
    return signals