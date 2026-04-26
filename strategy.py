#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrendFilter
Hypothesis: Elder Ray Bull/Bear Power (EMA13) with 1d EMA50 trend filter and ATR-based stop. Bull Power > 0 + Bear Power < 0 indicates balanced momentum; we enter when one power expands while the other contracts, aligned with 1d trend. Uses discrete sizing 0.25 to limit trades (~15-30/year). Works in bull/bear via 1d trend filter and momentum divergence logic.
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
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # ATR for volatility filtering and stoploss
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 50 for 1d EMA, 13 for EMA13, 14 for ATR
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(ema13[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_1d_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        atr_val = atr[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Bull market bias: Bull Power expanding (>0 and increasing) while Bear Power weakens (<0 and decreasing)
            # Bear market bias: Bear Power expanding (<0 and decreasing) while Bull Power weakens (>0 and increasing)
            bull_expanding = bull_val > 0 and bull_val > bull_power[i-1]
            bear_weakening = bear_val < 0 and bear_val < bear_power[i-1]
            bear_expanding = bear_val < 0 and bear_val < bear_power[i-1]
            bull_weakening = bull_val > 0 and bull_val > bull_power[i-1]
            
            # Long entry: bull power expanding in uptrend OR bear power weakening in uptrend
            long_entry = (bull_expanding or bear_weakening) and close_val > ema_50_val
            # Short entry: bear power expanding in downtrend OR bull power weakening in downtrend
            short_entry = (bear_expanding or bull_weakening) and close_val < ema_50_val
            
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
            # Long - exit on trend reversal or ATR-based stop
            # Exit if trend turns bearish OR price drops 2*ATR from entry
            if close_val < ema_50_val or close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal or ATR-based stop
            # Exit if trend turns bullish OR price rises 2*ATR from entry
            if close_val > ema_50_val or close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrendFilter"
timeframe = "6h"
leverage = 1.0