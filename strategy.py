#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1d ADX trend filter + volume spike
# Choppiness Index > 61.8 indicates ranging market (mean revert at Bollinger Bands)
# ADX > 25 indicates trending market (follow Donchian breakout)
# Volume > 2x 20-period average confirms breakout strength
# Designed for low trade frequency (<40/year) to minimize fee drag in choppy markets

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ADX trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX(14) for trend strength
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First TR has no previous close
    
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    plus_di14 = np.where(tr14 != 0, 100 * plus_dm14 / tr14, 0)
    minus_di14 = np.where(tr14 != 0, 100 * minus_dm14 / tr14, 0)
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d = adx  # Already 1d values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Load 1d data for Bollinger Bands (for chop regime)
    tp = (high_1d + low_1d + close_1d) / 3  # Typical price
    tp_ma = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    tp_std = pd.Series(tp).rolling(window=20, min_periods=20).std().values
    upper_bb = tp_ma + 2 * tp_std
    lower_bb = tp_ma - 2 * tp_std
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # 4h Choppiness Index (14-period)
    atr = np.abs(high - low)
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((hh - ll) != 0, 
                    100 * np.log10(atr_sum / (hh - ll)) / np.log10(14), 50)
    chop[np.isnan(chop)] = 50
    
    # 4h Donchian Channel (20) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(upper_bb_aligned[i]) or
            np.isnan(lower_bb_aligned[i]) or np.isnan(chop[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine market regime
            is_trending = adx_1d_aligned[i] > 25
            is_chopping = chop[i] > 61.8
            
            if is_trending:
                # Trending market: Donchian breakout with volume
                if (close[i] > highest_20[i-1] and  # breakout above
                    volume[i] > 2.0 * vol_avg_20[i]):  # volume spike
                    signals[i] = 0.30
                    position = 1
                elif (close[i] < lowest_20[i-1] and  # breakout below
                      volume[i] > 2.0 * vol_avg_20[i]):  # volume spike
                    signals[i] = -0.30
                    position = -1
            elif is_chopping:
                # Chopping market: mean reversion at Bollinger Bands
                if (close[i] < lower_bb_aligned[i] and  # at lower BB
                    volume[i] > 1.5 * vol_avg_20[i]):   # volume confirmation
                    signals[i] = 0.30
                    position = 1
                elif (close[i] > upper_bb_aligned[i] and  # at upper BB
                      volume[i] > 1.5 * vol_avg_20[i]):   # volume confirmation
                    signals[i] = -0.30
                    position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: opposite breakout or BB touch in chop
                if (adx_1d_aligned[i] > 25 and  # still trending
                    close[i] < lowest_20[i]):     # opposite Donchian break
                    signals[i] = 0.0
                    position = 0
                elif (chop[i] > 61.8 and        # chopping market
                      close[i] > upper_bb_aligned[i]):  # touched upper BB
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
            else:  # position == -1
                # Exit short: opposite breakout or BB touch in chop
                if (adx_1d_aligned[i] > 25 and  # still trending
                    close[i] > highest_20[i]):    # opposite Donchian break
                    signals[i] = 0.0
                    position = 0
                elif (chop[i] > 61.8 and        # chopping market
                      close[i] < lower_bb_aligned[i]):  # touched lower BB
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
    
    return signals

name = "4h_Chop_ADX_Regime_BB_Donchian"
timeframe = "4h"
leverage = 1.0