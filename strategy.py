#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Elder Ray (Bull/Bear Power) with 1d trend filter (EMA50)
# - Bull Power = High - EMA13(close), Bear Power = Low - EMA13(close) on 12h timeframe
# - Long when Bull Power > 0 AND price > 1d EMA50 (uptrend filter)
# - Short when Bear Power < 0 AND price < 1d EMA50 (downtrend filter)
# - Uses volume confirmation: current 6h volume > 1.5x 20-period average to filter low-volatility breakouts
# - Fixed position size 0.25 to manage drawdown
# - Elder Ray measures price strength relative to EMA; works in bull/bear via 1d EMA50 trend filter
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years)

name = "6h_12h_1d_elder_ray_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate EMA13 on 12h close for Elder Ray
    close_12h_series = pd.Series(close_12h)
    ema13_12h = close_12h_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_12h = high_12h - ema13_12h
    bear_power_12h = low_12h - ema13_12h
    
    # Align Elder Ray to 6h timeframe (wait for completed 12h bar)
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power_12h)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power_12h)
    
    # Load 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit long if Bull Power turns negative OR price breaks below 1d EMA50
            if bull_power_aligned[i] <= 0 or close[i] < ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if Bear Power turns positive OR price breaks above 1d EMA50
            if bear_power_aligned[i] >= 0 or close[i] > ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Elder Ray with volume confirmation and 1d trend filter
            if volume_confirmed:
                # Long entry: Bull Power positive AND price above 1d EMA50 (uptrend)
                if bull_power_aligned[i] > 0 and close[i] > ema50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Bear Power negative AND price below 1d EMA50 (downtrend)
                elif bear_power_aligned[i] < 0 and close[i] < ema50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals