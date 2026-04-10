#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter (EMA50) and volume confirmation
# - Long when price breaks above 20-bar Donchian high AND price > 1d EMA50 AND volume > 1.3x 20-bar avg
# - Short when price breaks below 20-bar Donchian low AND price < 1d EMA50 AND volume > 1.3x 20-bar avg
# - Exit when price crosses 10-bar Donchian midpoint (mean reversion or trend exhaustion)
# - Uses 1d EMA50 for trend filter to align with higher timeframe bias
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-30 trades/year on 12h timeframe (48-120 total over 4 years)
# - Donchian breakouts capture momentum; EMA filter avoids counter-trend trades; volume confirms conviction

name = "12h_1d_donchian_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period)
    highest_high = pd.Series(prices['high']).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(prices['low']).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0  # 10-bar midpoint for exit
    
    # Pre-compute volume confirmation: > 1.3x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.3 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute aligned 1d data
    c_1d = df_1d['close'].values
    c_1d_aligned = align_htf_to_ltf(prices, df_1d, c_1d)
    
    # Pre-compute 1d EMA(50) for trend filter
    ema50_1d = pd.Series(c_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    for i in range(50, n):  # Start after Donchian/EMA warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_20_avg[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(c_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian high AND in 1d uptrend with volume spike
            if (prices['close'].iloc[i] > highest_high[i] and 
                prices['close'].iloc[i] > ema50_1d_aligned[i] and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND in 1d downtrend with volume spike
            elif (prices['close'].iloc[i] < lowest_low[i] and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i] and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price crosses 10-bar Donchian midpoint (mean reversion)
            exit_signal = False
            if prices['close'].iloc[i] < donchian_mid[i]:
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