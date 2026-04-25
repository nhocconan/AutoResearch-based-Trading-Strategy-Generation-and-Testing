#!/usr/bin/env python3
"""
6h Elder Ray + Volume Spike with 1d EMA34 Trend
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) measures bull/bear strength relative to short-term trend.
In strong trends, Bull Power stays positive in uptrends and Bear Power negative in downtrends.
Combined with 1d EMA34 for higher-timeframe trend alignment and volume spikes for confirmation,
this strategy captures sustained momentum moves while avoiding choppy markets.
Volume spikes filter ensures entries occur with institutional participation.
6h timeframe targets 12-37 trades/year, minimizing fee drag.
Works in both bull (strong Bull Power + uptrend) and bear (strong Bear Power + downtrend) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA13 for Elder Ray (short-term trend)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(13, 20, 34)  # EMA13, volume MA, daily EMA alignment
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_volume = volume[i]
        vol_spike = volume_spike[i]
        
        # Daily trend filter: price above/below EMA34
        uptrend = ema_34_aligned[i] is not None and curr_close > ema_34_aligned[i]
        downtrend = ema_34_aligned[i] is not None and curr_close < ema_34_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: Bull Power > 0 (strong buying) AND uptrend AND volume spike
            long_entry = (curr_bull_power > 0) and uptrend and vol_spike
            # Short: Bear Power < 0 (strong selling) AND downtrend AND volume spike
            short_entry = (curr_bear_power < 0) and downtrend and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Bull Power turns negative OR loss of uptrend (price < EMA34)
            if (curr_bull_power <= 0) or (curr_close < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Bear Power turns positive OR loss of downtrend (price > EMA34)
            if (curr_bear_power >= 0) or (curr_close > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_VolumeSpike_1dEMA34_Trend"
timeframe = "6h"
leverage = 1.0