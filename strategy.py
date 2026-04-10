#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d trend filter and volume confirmation
# - Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Long when Bull Power > 0 AND 1d close > 1d EMA(50) AND 6h volume > 1.5x 20-bar avg
# - Short when Bear Power > 0 AND 1d close < 1d EMA(50) AND 6h volume > 1.5x 20-bar avg
# - Exit when power of current side <= 0 (momentum exhaustion)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Elder Ray measures bull/bear power relative to EMA; 1d EMA filter ensures alignment with daily trend
# - Volume confirmation avoids low-liquidity false signals
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: trend filter prevents counter-trend trades, power signals capture momentum

name = "6h_1d_elder_ray_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    close_gt_ema = close_1d > ema_50_1d
    close_lt_ema = close_1d < ema_50_1d
    
    # Align 1d trend to 6h timeframe
    close_gt_ema_aligned = align_htf_to_ltf(prices, df_1d, close_gt_ema)
    close_lt_ema_aligned = align_htf_to_ltf(prices, df_1d, close_lt_ema)
    
    # Pre-compute Elder Ray Index on 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Elder Ray conditions: power > 0 indicates bull/bear strength
    bull_power_positive = bull_power > 0
    bear_power_positive = bear_power > 0
    
    # Pre-compute 6h volume confirmation: > 1.5x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(close_gt_ema_aligned[i]) or np.isnan(close_lt_ema_aligned[i]) or
            np.isnan(bull_power_positive[i]) or np.isnan(bear_power_positive[i]) or
            np.isnan(vol_spike[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long when Bull Power > 0 AND 1d close > EMA(50) AND volume spike
            if (bull_power_positive[i] and 
                close_gt_ema_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when Bear Power > 0 AND 1d close < EMA(50) AND volume spike
            elif (bear_power_positive[i] and 
                  close_lt_ema_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when power <= 0 (momentum exhaustion)
            # Exit when power of current side <= 0
            if position == 1:
                exit_signal = bull_power_positive[i] == False  # Bull Power <= 0
            else:  # position == -1
                exit_signal = bear_power_positive[i] == False  # Bear Power <= 0
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals