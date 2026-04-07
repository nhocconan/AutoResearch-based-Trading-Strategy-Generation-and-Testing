#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Elder Ray Power with 1d EMA Trend Filter and Volume Confirmation
# Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) 
# measures bull/bear strength relative to trend. Combined with 1d EMA50 for trend filter
# and volume confirmation to ensure institutional participation. Works in both bull and bear 
# markets by only taking trades in direction of higher timeframe trend.
# Targets 15-35 trades/year with disciplined entries to avoid overtrading.

name = "6h_elder_ray_1d_trend_volume_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Elder Ray components: EMA13 of close
    ema13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = ema13 - low   # Bear Power = EMA13 - Low
    
    # 20-period average of Bull/Bear Power for threshold
    avg_bull_power = pd.Series(bull_power).rolling(window=20, min_periods=20).mean().values
    avg_bear_power = pd.Series(bear_power).rolling(window=20, min_periods=20).mean().values
    
    # 20-period SMA for volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if required data not available
        if (np.isnan(ema50_6h[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(avg_bull_power[i]) or 
            np.isnan(avg_bear_power[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: Bear Power becomes positive (losing bearish pressure) OR trend turns down
            if bear_power[i] > 0 or close[i] < ema50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: Bull Power becomes negative (losing bullish pressure) OR trend turns up
            if bull_power[i] < 0 or close[i] > ema50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: Strong Bull Power (below average indicates selling exhaustion) + volume confirmation + uptrend
            if (bull_power[i] < avg_bull_power[i] and  # Bull Power below its average = buying pressure building
                vol_confirm and 
                close[i] > ema50_6h[i]):
                position = 1
                signals[i] = 0.25
            # Short: Strong Bear Power (below average indicates buying exhaustion) + volume confirmation + downtrend
            elif (bear_power[i] < avg_bear_power[i] and  # Bear Power below its average = selling pressure building
                  vol_confirm and 
                  close[i] < ema50_6h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals