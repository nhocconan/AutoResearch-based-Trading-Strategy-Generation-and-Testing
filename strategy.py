#!/usr/bin/env python3
"""
6h Elder Ray + Volume Spike + Weekly Trend
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) measures 
buying/selling pressure relative to trend. In 6h timeframe, we take longs when 
Bull Power > 0 AND rising AND volume spike, shorts when Bear Power < 0 AND falling 
AND volume spike. Weekly EMA34 filter ensures we trade with the higher timeframe 
trend, working in both bull (long bias) and bear (short bias) markets. 
Target: 12-37 trades/year (50-150 over 4 years) with discrete sizing 0.25.
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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA34 for trend filter (needs extra delay - EMA confirmed on weekly close)
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w, additional_delay_bars=1)
    
    # 6h EMA13 for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Buying pressure above trend
    bear_power = low - ema_13   # Selling pressure below trend
    
    # 6h ATR(14) for volatility normalization
    tr1 = pd.Series(high).sub(pd.Series(low))
    tr2 = pd.Series(high).sub(pd.Series(close).shift(1)).abs()
    tr3 = pd.Series(low).sub(pd.Series(close).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(34, 20, 13) + 1  # Weekly EMA34 + volume MA + EMA13 + 1 for safety
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(ema_13[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_bull = bull_power[i]
        curr_bear = bear_power[i]
        vol_spike = volume_spike[i]
        
        # Weekly trend filter
        weekly_uptrend = ema_34_1w_aligned[i] is not None and curr_close > ema_34_1w_aligned[i]
        weekly_downtrend = ema_34_1w_aligned[i] is not None and curr_close < ema_34_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: Bull Power > 0 AND rising (bull power increasing) AND volume spike AND weekly uptrend
            long_entry = (curr_bull > 0) and (i > start_idx and curr_bull > bull_power[i-1]) and vol_spike and weekly_uptrend
            # Short: Bear Power < 0 AND falling (bear power decreasing) AND volume spike AND weekly downtrend
            short_entry = (curr_bear < 0) and (i > start_idx and curr_bear < bear_power[i-1]) and vol_spike and weekly_downtrend
            
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
            # Exit: Bull Power <= 0 (loss of buying pressure) OR weekly trend turns down
            if (curr_bull <= 0) or (not weekly_uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Bear Power >= 0 (loss of selling pressure) OR weekly trend turns up
            if (curr_bear >= 0) or (not weekly_downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_VolumeSpike_WeeklyEMA34_Trend"
timeframe = "6h"
leverage = 1.0