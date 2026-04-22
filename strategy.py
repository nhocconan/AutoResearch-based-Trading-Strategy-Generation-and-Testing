#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high and weekly trend is up.
# Short when price breaks below Donchian(20) low and weekly trend is down.
# Uses volume confirmation (1.5x 20-day average) to filter false breakouts.
# Designed for 1d timeframe to target 10-30 trades/year per symbol.
# Works in bull/bear via weekly trend filter to avoid counter-trend trades.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for higher timeframe trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1w EMA(50) for higher timeframe trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation (20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + weekly uptrend + volume confirmation
            if (close[i] > high_max[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + weekly downtrend + volume confirmation
            elif (close[i] < low_min[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Donchian break or trend reversal
            if position == 1:
                # Exit long on break below Donchian low or weekly trend reversal
                if (close[i] < low_min[i] or close[i] < ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short on break above Donchian high or weekly trend reversal
                if (close[i] > high_max[i] or close[i] > ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Trend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0