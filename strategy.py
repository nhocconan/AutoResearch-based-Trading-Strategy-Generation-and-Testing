#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volatility filter and volume confirmation
# Uses Donchian(20) channel breakouts for trend following, filtered by 1d ATR volatility regime
# and volume spike confirmation. Designed for low trade frequency (20-40/year) to minimize
# fee drag while capturing strong trending moves in both bull and bear markets.

name = "4h_Donchian20_Breakout_VolATR_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 10-period ATR on daily for volatility regime
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])
    
    atr_10 = np.full_like(close_1d, np.nan)
    for i in range(10, len(tr_1d)):
        atr_10[i] = np.nanmean(tr_1d[i-9:i+1])
    
    # Volatility filter: only trade when ATR is above its 50-period median (high volatility regime)
    atr_median = np.full_like(close_1d, np.nan)
    for i in range(50, len(atr_10)):
        valid_vals = atr_10[i-49:i+1]
        if np.sum(~np.isnan(valid_vals)) >= 20:  # Need reasonable sample
            atr_median[i] = np.nanmedian(valid_vals)
    
    vol_regime = atr_10 > atr_median  # True when volatility is high
    
    # Align volatility regime to 4h
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # Calculate Donchian channels (20-period) on 4h
    donchian_high = np.full_like(close, np.nan)
    donchian_low = np.full_like(close, np.nan)
    
    for i in range(20, len(high)):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation: 4h volume > 1.3x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above Donchian high with volatility and volume confirmation
            if (close[i] > donchian_high[i] and 
                vol_regime_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below Donchian low with volatility and volume confirmation
            elif (close[i] < donchian_low[i] and 
                  vol_regime_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low or volatility drops
            if close[i] < donchian_low[i] or not vol_regime_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high or volatility drops
            if close[i] > donchian_high[i] or not vol_regime_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals