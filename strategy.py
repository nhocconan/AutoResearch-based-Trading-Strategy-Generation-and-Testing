#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA(20) trend + volume confirmation.
# Uses weekly EMA for trend direction, daily Donchian breakout for entry.
# Volume spike filter reduces false signals.
# Long in uptrend when price breaks above Donchian upper band + volume spike.
# Short in downtrend when price breaks below Donchian lower band + volume spike.
# Designed to work in both bull and bear markets via trend-following breakouts.
# Target: 10-25 trades/year per symbol (40-100 total) to stay within fee limits.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for Donchian bands (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian(20) bands on daily data
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Load weekly data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(20) for trend direction
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume spike filter (20-period on daily volume)
    vol_1d = df_1d['volume'].values
    vol_ma20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_ma20 > 0  # Avoid division by zero
    vol_spike = vol_1d > 1.5 * vol_ma20
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: uptrend (close > weekly EMA20) + break above Donchian upper + volume spike
            if (close[i] > ema_20_1w_aligned[i] and 
                close[i] > upper_20_aligned[i] and 
                vol_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: downtrend (close < weekly EMA20) + break below Donchian lower + volume spike
            elif (close[i] < ema_20_1w_aligned[i] and 
                  close[i] < lower_20_aligned[i] and 
                  vol_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: trend reversal or Donchian opposite break
            if position == 1:
                if (close[i] < ema_20_1w_aligned[i] or 
                    close[i] < lower_20_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > ema_20_1w_aligned[i] or 
                    close[i] > upper_20_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA20_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0