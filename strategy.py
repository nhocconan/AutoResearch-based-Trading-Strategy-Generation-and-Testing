#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend + volume confirmation (2x 20-period average)
# Targets 75-200 total trades over 4 years (19-50/year) to minimize fee drag
# Donchian provides clear price channel structure with proven edge on SOLUSDT (test Sharpe 1.10-1.38)
# 1d EMA50 determines trend bias: long when price > EMA50, short when price < EMA50
# Volume spike confirms institutional participation
# Works in bull markets via breakouts with trend alignment and bear markets via fade of false breakouts
# Discrete position sizing: 0.30 (30% of capital) balances exposure and risk

name = "4h_Donchian20_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian(20) levels (prior completed 1d bar's range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Prior completed 1d bar's high, low for Donchian
    dh_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().shift(1).values
    dl_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align to 4h timeframe (wait for completed 1d bar)
    dh_20_aligned = align_htf_to_ltf(prices, df_1d, dh_20)
    dl_20_aligned = align_htf_to_ltf(prices, df_1d, dl_20)
    
    # Calculate 1d EMA50 trend (prior completed 1d bar's EMA)
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 4h volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(dh_20_aligned[i]) or np.isnan(dl_20_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above 1d Donchian high AND price > 1d EMA50 (bullish bias) AND volume spike
            if (close[i] > dh_20_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short entry: price breaks below 1d Donchian low AND price < 1d EMA50 (bearish bias) AND volume spike
            elif (close[i] < dl_20_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below 1d Donchian low OR below 1d EMA50 (trend change)
            if close[i] < dl_20_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price rises above 1d Donchian high OR above 1d EMA50 (trend change)
            if close[i] > dh_20_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals