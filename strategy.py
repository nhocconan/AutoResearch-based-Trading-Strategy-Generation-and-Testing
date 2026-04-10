#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Elder Ray Index with 12h volume confirmation and chop regime filter
# - Primary: 4h Elder Ray Bull Power (EMA13) > 0 for long, Bear Power (EMA13) < 0 for short
# - Volume filter: 12h volume > 1.3x 20-period volume MA to confirm institutional interest
# - Regime filter: Choppiness Index(14) > 61.8 (ranging market) for mean reversion to work
# - Exit: Elder Ray power crosses zero (momentum exhaustion)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Elder Ray shows bull/bear power, chop filter ensures mean reversion environment
# - Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe

name = "4h_12h_elder_ray_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    volume_12h = df_12h['volume'].values
    
    # Calculate EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Calculate Elder Ray Bull Power and Bear Power
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Calculate 12h volume confirmation: volume > 1.3x 20-period volume MA
    volume_ma_20_12h = pd.Series(volume_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    
    # Calculate 14-period Choppiness Index for regime filter (using 4h data)
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    
    # Handle first element
    high_low[0] = high[0] - low[0]
    high_close[0] = np.abs(high[0] - close[0])
    low_close[0] = np.abs(low[0] - close[0])
    
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_hl = max_high - min_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    chop = 100 * np.log10(atr_sum / range_hl) / np.log10(14)
    chop_filter = chop > 61.8  # Chop > 61.8 indicates ranging market (good for mean reversion)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_ma_20_12h_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Align 12h volume data for current bar
        volume_12h_current = align_htf_to_ltf(prices, df_12h, volume_12h)
        vol_confirm = volume_12h_current[i] > 1.3 * volume_ma_20_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 (bullish momentum) + vol confirmation + chop filter
            if (bull_power[i] > 0 and 
                vol_confirm and chop_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power < 0 (bearish momentum) + vol confirmation + chop filter
            elif (bear_power[i] < 0 and 
                  vol_confirm and chop_filter[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when power crosses zero (momentum exhaustion)
            # Exit: Elder Ray power crosses zero
            if position == 1:  # Long position
                if bull_power[i] <= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if bear_power[i] >= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals