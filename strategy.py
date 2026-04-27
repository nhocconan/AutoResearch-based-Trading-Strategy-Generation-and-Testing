#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike_ATRStop
Hypothesis: Donchian(20) breakout on 4h with 12h EMA50 trend filter and volume confirmation.
Works in bull/bear markets: In trending regimes (price > EMA50 for longs, < EMA50 for shorts),
Donchian breakouts with volume capture momentum. Uses ATR-based stoploss and discrete sizing (0.25)
to limit drawdown and reduce fee churn. Targets 50-150 trades over 4 years on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA50 trend filter
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Donchian(20) channels from 4h data
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # ATR(14) for stoploss and volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike: current > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position
    
    # Warmup: need EMA50, Donchian, ATR, vol avg
    start_idx = max(50, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(atr[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper_val = upper[i]
        lower_val = lower[i]
        ema_val = ema_50_aligned[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with EMA alignment and volume spike
            long_condition = (close_val > upper_val and 
                            close_val > ema_val and 
                            vol_spike)
            short_condition = (close_val < lower_val and 
                             close_val < ema_val and 
                             vol_spike)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit long: Donchian lower band breach OR ATR stoploss
            if close_val < lower_val or close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Donchian upper band breach OR ATR stoploss
            if close_val > upper_val or close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0