#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Volume Spike + 1d Trend Filter
# - Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# - Long when Bull Power > 0 AND Bear Power rising (less negative) AND volume > 1.5x average AND 1d close > 1d EMA50
# - Short when Bear Power < 0 AND Bull Power falling (less positive) AND volume > 1.5x average AND 1d close < 1d EMA50
# - Exit when Elder Ray signal reverses OR volume drops below 0.7x average
# - Uses 1d EMA50 trend filter to avoid counter-trend trades
# - Volume confirmation filters weak breakouts
# - Target: 12-25 trades/year (50-100 total over 4 years) to minimize fee drag
# - Works in bull (strong Bull Power) and bear (strong Bear Power) regimes
# - Elder Ray measures trend strength via price relative to EMA, effective in both regimes

name = "6h_1d_elder_ray_volume_trend_v1"
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
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute Elder Ray components on 6h data
    ema13 = pd.Series(prices['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = prices['high'].values - ema13  # High - EMA13
    bear_power = prices['low'].values - ema13   # Low - EMA13
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.7x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.7 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 AND Bear Power rising (less negative than previous) 
            #            AND volume spike AND 1d uptrend
            if (bull_power[i] > 0 and 
                i > 0 and bear_power[i] > bear_power[i-1] and  # Bear Power rising
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power < 0 AND Bull Power falling (less positive than previous)
            #            AND volume spike AND 1d downtrend
            elif (bear_power[i] < 0 and 
                  i > 0 and bull_power[i] < bull_power[i-1] and  # Bull Power falling
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Elder Ray signal reverses
            # 2. Volume drops below 0.7x average (loss of momentum)
            if position == 1:  # Long position
                long_exit = (bull_power[i] <= 0) or vol_weak.iloc[i]
                if long_exit:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                short_exit = (bear_power[i] >= 0) or vol_weak.iloc[i]
                if short_exit:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals