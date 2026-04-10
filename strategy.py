#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d trend filter and volume confirmation
# - Long when price breaks above 20-period Donchian upper band AND 1d EMA50 rising AND volume > 1.5x 20-bar avg
# - Short when price breaks below 20-period Donchian lower band AND 1d EMA50 falling AND volume > 1.5x 20-bar avg
# - Exit when price crosses the Donchian midline (mean reversion to equilibrium)
# - Uses 1d EMA50 for trend filter to avoid counter-trend trades
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Donchian breakouts capture strong momentum moves; trend filter adds directional bias

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
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute Donchian channels (20-period) from 12h data
    high_12h = prices['high'].rolling(window=20, min_periods=20).max().values
    low_12h = prices['low'].rolling(window=20, min_periods=20).min().values
    upper_donch = high_12h
    lower_donch = low_12h
    mid_donch = (upper_donch + lower_donch) / 2.0
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(upper_donch[i]) or 
            np.isnan(lower_donch[i]) or np.isnan(mid_donch[i]) or 
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
            # Long when price breaks above upper Donchian AND 1d uptrend with volume spike
            if (prices['close'].iloc[i] > upper_donch[i] and 
                prices['close'].iloc[i] > ema50_1d_aligned[i] and  # price above 1d EMA50
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below lower Donchian AND 1d downtrend with volume spike
            elif (prices['close'].iloc[i] < lower_donch[i] and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i] and  # price below 1d EMA50
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to midline (mean reversion)
            # Exit when price crosses the Donchian midline
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] <= mid_donch[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] >= mid_donch[i]:
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