#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (EMA50 slope) and volume confirmation (>1.5x 20-bar avg)
# - Long when price breaks above Donchian high(20) AND 1d EMA50 rising AND volume > 1.5x 20-bar avg
# - Short when price breaks below Donchian low(20) AND 1d EMA50 falling AND volume > 1.5x 20-bar avg
# - Exit when price returns to Donchian midpoint (mean reversion to equilibrium)
# - Uses 1d EMA50 slope for trend filter to avoid counter-trend trades
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Donchian breakouts capture momentum; trend filter adds directional bias; volume confirms conviction

name = "4h_1d_donchian_breakout_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period) from 4h data
    high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2
    
    # Pre-compute 1d EMA(50) and its slope for trend filter
    close_1d_arr = df_1d['close'].values
    ema50_1d = pd.Series(close_1d_arr).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_slope = ema50_1d - np.roll(ema50_1d, 1)  # daily change
    ema50_1d_slope[0] = 0  # first value has no prior
    
    # Align HTF EMA50 and slope to LTF
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema50_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d_slope)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_1d_slope_aligned[i]) or
            np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian high AND 1d uptrend (rising EMA50) with volume spike
            if (prices['close'].iloc[i] > high_20[i] and 
                ema50_1d_slope_aligned[i] > 0 and  # 1d EMA50 rising
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND 1d downtrend (falling EMA50) with volume spike
            elif (prices['close'].iloc[i] < low_20[i] and 
                  ema50_1d_slope_aligned[i] < 0 and  # 1d EMA50 falling
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Donchian midpoint (mean reversion)
            # Exit when price returns to Donchian midpoint
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] <= donchian_mid[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] >= donchian_mid[i]:
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