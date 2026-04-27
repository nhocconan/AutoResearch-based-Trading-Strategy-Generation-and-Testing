#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_Regime_New
Hypothesis: Uses 4h timeframe with Camarilla R1/S1 breakouts filtered by 12h EMA50 trend, volume confirmation, and chop regime filter. Designed for BTC/ETH to work in both bull and bear markets by only taking breakouts in the direction of the 12h trend. Target 20-50 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA50 trend filter
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Camarilla levels from previous completed 1d bar
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    rng = prev_high - prev_low
    r1 = prev_close + (rng * 1.1 / 12)
    s1 = prev_close - (rng * 1.1 / 12)
    
    # Align Camarilla levels to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 1.8 * 20-period average (moderate threshold)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_avg)
    
    # Choppiness regime filter: CHOP(14) > 61.8 = range (mean revert), CHOP < 38.2 = trending (trend follow)
    # We'll use trending regime only (CHOP < 38.2) to trade with the trend
    hl_range = pd.Series(high - low).rolling(window=14, min_periods=14).sum().values
    true_range = pd.Series(np.maximum(
        high - low,
        np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    ), index=prices.index).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(hl_range / true_range) / np.log10(14)
    chop_aligned = chop  # already LTF
    chop_filter = chop_aligned < 38.2  # trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # Discrete size to minimize fee churn
    
    # Warmup: need 12h EMA50 (50), 1d shift(1) for Camarilla, vol avg (20), chop (14)
    start_idx = max(50 + 2*6, 1 + 2*6, 20, 14)  # ~112 bars for 12h EMA50 warmup (12h bars per day = 2)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_conf = volume_confirm[i]
        chop_regime = chop_filter[i]
        
        if position == 0:
            # Look for entry: Camarilla R1/S1 breakout with 12h EMA50 alignment, volume confirmation, and trending regime
            long_condition = (close_val > r1_val and 
                            close_val > ema_val and 
                            vol_conf and 
                            chop_regime)
            short_condition = (close_val < s1_val and 
                             close_val < ema_val and 
                             vol_conf and 
                             chop_regime)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit long: price crosses below 12h EMA50 (trend reversal)
            if close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above 12h EMA50 (trend reversal)
            if close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_Regime_New"
timeframe = "4h"
leverage = 1.0