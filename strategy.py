#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA34 trend filter and volume confirmation
# Uses Donchian channel breakouts for trend continuation, filtered by 12h EMA trend
# Long when price breaks above 20-period high with 12h uptrend and volume spike
# Short when price breaks below 20-period low with 12h downtrend and volume spike
# Designed for 6h timeframe to target 15-25 trades/year per symbol.
# Donchian breakouts work well in trending markets; EMA filter reduces whipsaws in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # 12h EMA(34) for higher timeframe trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Donchian channel (20-period) on 6h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period on 6h data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 20-period high + 12h uptrend + volume spike
            if (close[i] > high_20[i] and 
                close[i] > ema_34_12h_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low + 12h downtrend + volume spike
            elif (close[i] < low_20[i] and 
                  close[i] < ema_34_12h_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to 20-period midpoint or trend reversal
            mid_20 = (high_20[i] + low_20[i]) / 2.0
            
            if position == 1:
                # Exit on price below midpoint or trend reversal
                if (close[i] < mid_20 or 
                    close[i] < ema_34_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on price above midpoint or trend reversal
                if (close[i] > mid_20 or 
                    close[i] > ema_34_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_12hEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0