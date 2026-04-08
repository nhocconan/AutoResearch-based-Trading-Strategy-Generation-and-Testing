#!/usr/bin/env python3
# 4h_volatility_breakout_volume_regime_v1
# Hypothesis: 4h ATR-based volatility breakout with volume confirmation and
# 12h choppiness regime filter. Works in bull/bear by adapting to volatility
# regimes - breakouts work in trending markets, while the chop filter avoids
# false signals in ranging markets. Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_volatility_breakout_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR(14) for volatility breakout
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.nanmean(tr[i-14:i+1])
    
    # Donchian channel (20-period) for breakout levels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i+1])
        donchian_low[i] = np.min(low[i-20:i+1])
    
    # Volume moving average (20-period) for confirmation
    vol_ma = np.full(n, np.nan)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for choppiness regime filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # ATR for 12h chop calculation
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr2_12h[0] = np.nan
    tr3_12h[0] = np.nan
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    
    atr_12h = np.full(len(df_12h), np.nan)
    for i in range(14, len(df_12h)):
        atr_12h[i] = np.nanmean(tr_12h[i-14:i+1])
    
    # True range sum and ATR sum for choppiness
    tr_sum_12h = np.full(len(df_12h), np.nan)
    atr_sum_12h = np.full(len(df_12h), np.nan)
    for i in range(14, len(df_12h)):
        tr_sum_12h[i] = np.nansum(tr_12h[i-14:i+1])
        atr_sum_12h[i] = np.nansum(atr_12h[i-14:i+1]) if not np.isnan(atr_12h[i-14:i+1]).all() else np.nan
    
    # Choppiness index: 100 * log10(atr_sum / tr_sum) / log10(period)
    chop_12h = np.full(len(df_12h), np.nan)
    for i in range(14, len(df_12h)):
        if tr_sum_12h[i] > 0 and atr_sum_12h[i] > 0 and not np.isnan(tr_sum_12h[i]) and not np.isnan(atr_sum_12h[i]):
            chop_12h[i] = 100 * np.log10(atr_sum_12h[i] / tr_sum_12h[i]) / np.log10(14)
        else:
            chop_12h[i] = np.nan
    
    # Align 12h chop to 4h
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(20, 14)  # Donchian and ATR ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(chop_12h_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Chop regime filter: only trade when chop < 50 (trending market)
        trending_regime = chop_12h_aligned[i] < 50
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or volatility drops
            if close[i] < donchian_low[i] or atr[i] < 0.5 * np.nanmean(atr[max(0, i-10):i+1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or volatility drops
            if close[i] > donchian_high[i] or atr[i] < 0.5 * np.nanmean(atr[max(0, i-10):i+1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with volume confirmation in trending regime
            if (close[i] > donchian_high[i] and 
                volume_confirm and 
                trending_regime):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume confirmation in trending regime
            elif (close[i] < donchian_low[i] and 
                  volume_confirm and 
                  trending_regime):
                position = -1
                signals[i] = -0.25
    
    return signals