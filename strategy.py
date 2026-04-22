#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA(20) trend filter and volume confirmation.
# Uses weekly EMA(20) for trend direction, daily Donchian(20) for breakout entry.
# Volume spike filter reduces false signals.
# Long in uptrend when price breaks above Donchian upper band + volume spike.
# Short in downtrend when price breaks below Donchian lower band + volume spike.
# Designed to work in both bull and bear markets via trend-following breakout entries.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years) to stay within fee limits.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Donchian bands (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian(20) bands on 1d
    high_roll = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_upper = align_htf_to_ltf(prices, df_1d, high_roll)
    donchian_lower = align_htf_to_ltf(prices, df_1d, low_roll)
    
    # Load 1w data for EMA(20) trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA(20) for trend direction
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume spike filter (20-period on 1d volume)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: uptrend (close > EMA20) + breakout above upper band + volume spike
            if (close[i] > ema_20_1w_aligned[i] and 
                close[i] > donchian_upper[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: downtrend (close < EMA20) + breakout below lower band + volume spike
            elif (close[i] < ema_20_1w_aligned[i] and 
                  close[i] < donchian_lower[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: trend reversal or opposite breakout
            if position == 1:
                if (close[i] < ema_20_1w_aligned[i] or close[i] < donchian_lower[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > ema_20_1w_aligned[i] or close[i] > donchian_upper[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA20_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0