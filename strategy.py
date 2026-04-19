#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian breakout with weekly EMA50 filter and volume confirmation.
# Long when: Price breaks above Donchian(20) high, weekly EMA50 upward, volume > 1.5x 20-period average
# Short when: Price breaks below Donchian(20) low, weekly EMA50 downward, volume > 1.5x 20-period average
# Exit when: Price crosses back through the Donchian median (20-period midpoint)
# Donchian channels capture breakout momentum, EMA50 filters trend, volume confirms breakout strength.
# Target: 12-25 trades/year per symbol. Works in bull (buy breakouts) and bear (sell breakdowns).
name = "12h_Donchian20_WeeklyEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for EMA50
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to 12H timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Donchian channel parameters (20-period)
    donchian_period = 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = donchian_period  # Wait for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Calculate Donchian high and low for current bar
        start_lookback = i - donchian_period + 1
        if start_lookback < 0:
            signals[i] = 0.0
            continue
            
        donchian_high = np.max(high[start_lookback:i+1])
        donchian_low = np.min(low[start_lookback:i+1])
        donchian_mid = (donchian_high + donchian_low) / 2.0
        
        price = close[i]
        ema50 = ema50_1w_aligned[i]
        vol = volume[i]
        
        # Calculate 20-period volume average
        vol_start = max(0, i - 19)
        vol_ma_20 = np.mean(volume[vol_start:i+1]) if (i - vol_start + 1) >= 20 else np.nan
        
        if np.isnan(vol_ma_20):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: Price breaks above Donchian high, EMA50 upward, volume spike
            if (price > donchian_high and close[i-1] <= donchian_high and 
                ema50 > ema50_1w_aligned[i-1] and vol > 1.5 * vol_ma_20):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below Donchian low, EMA50 downward, volume spike
            elif (price < donchian_low and close[i-1] >= donchian_low and 
                  ema50 < ema50_1w_aligned[i-1] and vol > 1.5 * vol_ma_20):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses back below Donchian median
            if price < donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses back above Donchian median
            if price > donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals