#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_WeeklyTrend_VolumeSpike_ATRStop
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike (ATR ratio > 1.5). Trade only breakouts aligned with weekly trend during volatility expansion. Uses discrete sizing 0.25 to limit trades (~15/year). Volume spike ensures institutional participation. ATR-based stoploss manages risk. Works in bull/bear via trend filter and volatility regime.
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
    
    # Load weekly data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for stoploss and volume regime
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period ATR) for volume regime
    atr_ratio = atr / pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate Donchian(20) channels
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of calculations (50 for ATR ratio and EMA, 20 for Donchian)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_spike = atr_ratio[i] > 1.5  # volume spike
        atr_val = atr[i]
        size = fixed_size
        
        # Entry conditions: Donchian breakout with volume spike AND aligned with 1w EMA50 trend
        long_entry = (close_val > upper) and vol_spike and (close_val > ema_50_val)
        short_entry = (close_val < lower) and vol_spike and (close_val < ema_50_val)
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on Donchian lower band or ATR stoploss
            if close_val < lower:  # Donchian breakout reversal
                signals[i] = 0.0
                position = 0
            elif close_val < entry_price - 2.5 * atr_val:  # ATR stoploss
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on Donchian upper band or ATR stoploss
            if close_val > upper:  # Donchian breakout reversal
                signals[i] = 0.0
                position = 0
            elif close_val > entry_price + 2.5 * atr_val:  # ATR stoploss
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_Breakout_WeeklyTrend_VolumeSpike_ATRStop"
timeframe = "1d"
leverage = 1.0