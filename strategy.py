# 4h_HTF_Breaker_Block_Demand_Supply
# Hypothesis: Identify supply/demand zones on 1d (via swing highs/lows) and trade breakouts on 4h
# when price breaks above/below these zones with volume confirmation. Uses 1d swing points as
# dynamic support/resistance. Works in both bull and bear markets by trading breaks of
# institutional levels. Target: 20-40 trades/year.

name = "4h_HTF_Breaker_Block_Demand_Supply"
timeframe = "4h"
leverage = 1.0

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
    
    # 1d swing points for supply/demand zones
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Find swing highs and lows on 1d (3-bar lookback/forward)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    swing_high = np.zeros(len(high_1d), dtype=bool)
    swing_low = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            swing_high[i] = True
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            swing_low[i] = True
    
    # Create supply (from swing highs) and demand (from swing lows) zones
    supply_zones = np.full(len(high_1d), np.nan)
    demand_zones = np.full(len(low_1d), np.nan)
    
    for i in range(len(high_1d)):
        if swing_high[i]:
            supply_zones[i] = high_1d[i]  # Supply at swing high
        if swing_low[i]:
            demand_zones[i] = low_1d[i]   # Demand at swing low
    
    # Forward fill zones to create continuous levels
    supply_zones = pd.Series(supply_zones).ffill().bfill().values
    demand_zones = pd.Series(demand_zones).ffill().bfill().values
    
    # Align to 4h timeframe
    supply_zones_aligned = align_htf_to_ltf(prices, df_1d, supply_zones)
    demand_zones_aligned = align_htf_to_ltf(prices, df_1d, demand_zones)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(supply_zones_aligned[i]) or np.isnan(demand_zones_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        current_price = close[i]
        supply_level = supply_zones_aligned[i]
        demand_level = demand_zones_aligned[i]
        
        if position == 0:
            # Long: Break above supply zone with volume
            if current_price > supply_level and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below demand zone with volume
            elif current_price < demand_level and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long: Exit when price breaks below demand zone (support)
            if current_price < demand_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short: Exit when price breaks above supply zone (resistance)
            if current_price > supply_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals