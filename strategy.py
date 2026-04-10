#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with 12h trend filter and volume confirmation
# - Primary: 6h timeframe for moderate frequency and reduced fee drag
# - HTF: 12h for trend direction (EMA50 slope) and 1d for volume spike detection
# - Long: Bull Power > 0 (close > EMA13) AND Bear Power < 0 (open < EMA13) + 12h EMA50 rising + 1d volume > 1.5x 20-period MA
# - Short: Bear Power < 0 (open < EMA13) AND Bull Power < 0 (close < EMA13) + 12h EMA50 falling + 1d volume > 1.5x 20-period MA
# - Exit: Power signals reverse or volume drops below average
# - Position sizing: 0.25 (discrete level)
# - Target: 60-150 total trades over 4 years (15-38/year) - within 6h sweet spot
# - Works in bull/bear: Elder Ray captures institutional buying/selling pressure; volume filter avoids false signals in low-participation markets

name = "6h_12h_1d_elder_ray_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pre-compute 1d data
    volume_1d = df_1d['volume'].values
    
    # Calculate 6h EMA13 for Elder Ray
    close_6h_series = pd.Series(close_6h)
    ema13_6h = close_6h_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray Power (6h)
    bull_power = close_6h - ema13_6h  # Bull Power = Close - EMA13
    bear_power = open_6h - ema13_6h   # Bear Power = Open - EMA13
    
    # Calculate 12h EMA50 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    # Calculate EMA50 slope (rising/falling) - 3-period change
    ema50_slope = np.zeros_like(ema50_12h)
    ema50_slope[3:] = ema50_12h[3:] - ema50_12h[:-3]
    ema50_slope_aligned = align_htf_to_ltf(prices, df_12h, ema50_slope)
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_slope_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_spike = volume_1d[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 AND Bear Power < 0 (strong buying pressure) + rising trend + volume spike
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                ema50_slope_aligned[i] > 0 and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power < 0 AND Bull Power < 0 (strong selling pressure) + falling trend + volume spike
            elif (bear_power[i] < 0 and bull_power[i] < 0 and 
                  ema50_slope_aligned[i] < 0 and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Power signals reverse (loss of momentum)
            # 2. Volume drops below average (loss of participation)
            
            volume_normal = volume_1d[i] <= volume_ma_20_1d_aligned[i]
            
            if position == 1:  # Long position
                exit_condition = (
                    bull_power[i] <= 0 or      # Lost buying pressure
                    bear_power[i] >= 0 or      # Gained selling pressure
                    volume_normal              # Volume dried up
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    bear_power[i] >= 0 or      # Lost selling pressure
                    bull_power[i] >= 0 or      # Gained buying pressure
                    volume_normal              # Volume dried up
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals