#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h EMA trend filter and volume confirmation
# Long when price breaks above Donchian(20) high + price > 12h EMA50 + volume > 1.5x 20-period average
# Short when price breaks below Donchian(20) low + price < 12h EMA50 + volume > 1.5x 20-period average
# Exit when price crosses back through Donchian midline or trend reverses
# Designed for 4h timeframe to target 20-30 trades/year per symbol.
# Works in bull markets (captures breakouts) and bear markets (avoids false breaks via trend filter)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for trend filter (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA(50) for higher timeframe trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2
    
    # Volume confirmation (20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above Donchian high + uptrend + volume confirmation
            if (close[i] > high_20[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low + downtrend + volume confirmation
            elif (close[i] < low_20[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price crosses Donchian midline or trend reversal
            if position == 1:
                # Exit on price below midline or trend reversal
                if (close[i] < donchian_mid[i] or 
                    close[i] < ema_50_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on price above midline or trend reversal
                if (close[i] > donchian_mid[i] or 
                    close[i] > ema_50_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0