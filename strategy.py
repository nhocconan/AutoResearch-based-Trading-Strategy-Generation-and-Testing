#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d EMA200 trend filter and 1w pivot confirmation
# - Long when price breaks above Donchian(20) high AND price > 1d EMA200 AND price > 1w weekly pivot
# - Short when price breaks below Donchian(20) low AND price < 1d EMA200 AND price < 1w weekly pivot
# - Exit when price crosses 1d EMA200 (trend change)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years)
# - Donchian breakouts capture momentum; EMA200 filter avoids counter-trend trades; weekly pivot adds structural bias
# - Works in both bull (breakouts with trend) and bear (failed breaks reverse) markets

name = "6h_1d_1w_donchian_breakout_trend_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute Donchian(20) channels
    highest_high = pd.Series(prices['high']).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(prices['low']).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute aligned 1d EMA(200) for trend filter
    c_1d = df_1d['close'].values
    ema200_1d = pd.Series(c_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Pre-compute 1w weekly pivot (standard calculation: (H+L+C)/3)
    typical_price_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    pivot_1w = typical_price_1w.values
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_20_avg[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(pivot_1w_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian high AND in 1d uptrend AND above weekly pivot with volume spike
            if (prices['close'].iloc[i] > highest_high[i] and 
                prices['close'].iloc[i] > ema200_1d_aligned[i] and 
                prices['close'].iloc[i] > pivot_1w_aligned[i] and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND in 1d downtrend AND below weekly pivot with volume spike
            elif (prices['close'].iloc[i] < lowest_low[i] and 
                  prices['close'].iloc[i] < ema200_1d_aligned[i] and 
                  prices['close'].iloc[i] < pivot_1w_aligned[i] and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price crosses 1d EMA200 (trend change)
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] < ema200_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] > ema200_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals