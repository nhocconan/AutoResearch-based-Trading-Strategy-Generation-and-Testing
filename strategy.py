#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA (13-period) to assess trend strength
# 1w EMA50 filters for higher timeframe trend alignment (avoid counter-trend trades)
# Volume > 1.5x 20-period average confirms institutional participation
# Designed for 6h timeframe with selective entries to avoid overtrading
# Target: 12-37 trades per year per symbol (50-150 total over 4 years)
# Works in bull markets via bull power strength, in bear via bear power reversals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 50-period EMA on 1w timeframe for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Elder Ray components (13-period EMA)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 13-period EMA for Elder Ray basis
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(ema50_1w_aligned[i]) or \
           np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1w trend
        is_uptrend = close[i] > ema50_1w_aligned[i]
        is_downtrend = close[i] < ema50_1w_aligned[i]
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        price = close[i]
        
        if position == 0:
            # Long entry: Bull Power > 0 (strong bullish momentum) + uptrend + volume
            long_signal = (bull_power[i] > 0) and is_uptrend and has_volume
            
            # Short entry: Bear Power < 0 (strong bearish momentum) + downtrend + volume
            short_signal = (bear_power[i] < 0) and is_downtrend and has_volume
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: Bull Power turns negative (momentum fading)
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power turns positive (momentum fading)
            if bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1wTrendFilter_Volume"
timeframe = "6h"
leverage = 1.0