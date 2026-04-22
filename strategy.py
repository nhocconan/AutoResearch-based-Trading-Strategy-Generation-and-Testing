#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA10 trend filter and volume confirmation
# Long when price breaks above Donchian high (20) + 1w uptrend (price > EMA10) + volume spike
# Short when price breaks below Donchian low (20) + 1w downtrend (price < EMA10) + volume spike
# Uses weekly trend filter to avoid counter-trend trades in ranging markets
# Volume spike ensures breakouts have conviction
# Designed for daily timeframe to target 10-25 trades/year per symbol.
# Works in bull markets by catching breakouts and in bear markets by avoiding false breakdowns via trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Donchian channels (primary timeframe data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Load 1w data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Donchian channels (20-period) on 1d data
    # Using rolling window with min_periods
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1w EMA(10) for higher timeframe trend filter
    ema_10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Volume spike filter (20-period on 1d data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_10_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high + 1w uptrend + volume spike
            if (close[i] > high_max[i] and 
                close[i] > ema_10_1w_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + 1w downtrend + volume spike
            elif (close[i] < low_min[i] and 
                  close[i] < ema_10_1w_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price crosses Donchian midpoint or trend reversal
            mid = (high_max[i] + low_min[i]) / 2.0
            if position == 1:
                # Exit on price below midpoint or trend reversal
                if (close[i] < mid or 
                    close[i] < ema_10_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on price above midpoint or trend reversal
                if (close[i] > mid or 
                    close[i] > ema_10_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA10_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0