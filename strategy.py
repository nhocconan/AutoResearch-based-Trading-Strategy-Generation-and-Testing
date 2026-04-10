#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# - Long when price breaks above 20-period high AND 12h EMA50 rising AND volume > 2.0x 20-bar avg
# - Short when price breaks below 20-period low AND 12h EMA50 falling AND volume > 2.0x 20-bar avg
# - Exit when price crosses below 12h EMA50 (for long) or above 12h EMA50 (for short)
# - Uses 12h EMA50 for trend filter to avoid counter-trend trades
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 20-30 trades/year on 4h timeframe (80-120 total over 4 years)
# - Donchian breakouts capture strong momentum; trend filter improves win rate in bear markets

name = "4h_12h_donchian_breakout_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period high/low)
    high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Pre-compute volume confirmation: > 2.0x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (2.0 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above 20-period high AND 12h uptrend with volume spike
            if (prices['close'].iloc[i] > high_20[i] and 
                close_12h[-1] > ema50_12h[-1] and  # 12h close above EMA50 (uptrend)
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below 20-period low AND 12h downtrend with volume spike
            elif (prices['close'].iloc[i] < low_20[i] and 
                  close_12h[-1] < ema50_12h[-1] and  # 12h close below EMA50 (downtrend)
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when price crosses 12h EMA50
            # Exit when price crosses below 12h EMA50 (for long) or above 12h EMA50 (for short)
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] < ema50_12h_aligned[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] > ema50_12h_aligned[i]:
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