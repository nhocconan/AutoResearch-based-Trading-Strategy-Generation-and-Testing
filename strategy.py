#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with daily volume confirmation and 1w trend filter
# - Long when price breaks above 20-bar Donchian high AND daily volume > 1.5x 20-day avg AND 1w EMA50 rising
# - Short when price breaks below 20-bar Donchian low AND daily volume > 1.5x 20-day avg AND 1w EMA50 falling
# - Exit when price returns to opposite Donchian level (mean reversion to channel midpoint)
# - Uses 1w EMA50 for trend filter to avoid counter-trend trades in bear markets
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Donchian breakouts capture strong moves; volume confirmation filters false breakouts; weekly trend avoids bear traps

name = "12h_1d_1w_donchian_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period) from 12h data
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    
    # Donchian high/low: rolling max/min of high/low
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2  # midpoint for exit
    
    # Pre-compute daily volume confirmation: > 1.5x 20-day average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup (20 for Donchian + buffer)
        # Skip if any required data is invalid
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or np.isnan(ema50_1w_aligned[i]) or 
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
            # Long when price breaks above Donchian high AND volume spike AND 1w uptrend
            if (prices['high'].iloc[i] > donch_high[i] and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema50_1w_aligned[i]):  # price above 1w EMA50 for uptrend
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND volume spike AND 1w downtrend
            elif (prices['low'].iloc[i] < donch_low[i] and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema50_1w_aligned[i]):  # price below 1w EMA50 for downtrend
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to midpoint (mean reversion)
            # Exit when price returns to Donchian midpoint
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] <= donch_mid[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] >= donch_mid[i]:
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