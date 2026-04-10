#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# - Long when price breaks above Donchian(20) high in 1d uptrend (close > weekly EMA50) with volume > 2.0x 20-bar avg
# - Short when price breaks below Donchian(20) low in 1d downtrend (close < weekly EMA50) with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~15 trades/year (60 total over 4 years) to avoid fee drag
# - Weekly trend filter reduces false signals in counter-trend markets
# - Donchian breakout captures strong momentum moves in both bull and bear markets
# - Volume confirmation ensures institutional participation

name = "1d_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 1d indicators
    highest_high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    lowest_low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    
    # 1d volume confirmation: > 2.0x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'].values > (2.0 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: Donchian breakout above in 1d uptrend with volume spike
            if (prices['close'].iloc[i] > highest_high_20[i] and 
                prices['close'].iloc[i] > ema_50_1w_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: Donchian breakdown below in 1d downtrend with volume spike
            elif (prices['close'].iloc[i] < lowest_low_20[i] and 
                  prices['close'].iloc[i] < ema_50_1w_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit when price returns to opposite Donchian level
            if position == 1 and prices['close'].iloc[i] < lowest_low_20[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and prices['close'].iloc[i] > highest_high_20[i]:
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals