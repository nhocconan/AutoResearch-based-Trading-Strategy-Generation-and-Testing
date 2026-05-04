#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation
# Donchian breakout captures strong momentum, EMA50 ensures trend alignment, volume spike confirms conviction
# Targets 12-30 trades/year (50-120 total) to minimize fee drag while maintaining edge in bull/bear markets
# Uses discrete position sizing (0.0, ±0.25) to reduce churn

name = "12h_Donchian20_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter from prior completed 1d bar
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_shifted = np.roll(ema50_1d, 1)
    ema50_1d_shifted[0] = np.nan
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d_shifted)
    
    # Calculate 12h Donchian channels (20-period)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Calculate Donchian channels for 12h timeframe using available data
        period = 20
        if i >= period:
            highest_high = np.max(high[i-period+1:i+1])
            lowest_low = np.min(low[i-period+1:i+1])
            
            # Skip if any value is NaN
            if (np.isnan(highest_high) or np.isnan(lowest_low) or 
                np.isnan(ema50_1d_aligned[i])):
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                continue
            
            if position == 0:
                # Long conditions: break above Donchian high AND 1d EMA50 uptrend AND volume spike
                if close[i] > highest_high and close[i] > ema50_1d_aligned[i] and volume[i] > (2.0 * np.mean(volume[max(0,i-19):i+1])):
                    signals[i] = 0.25
                    position = 1
                # Short conditions: break below Donchian low AND 1d EMA50 downtrend AND volume spike
                elif close[i] < lowest_low and close[i] < ema50_1d_aligned[i] and volume[i] > (2.0 * np.mean(volume[max(0,i-19):i+1])):
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Exit long: price closes below Donchian low OR below 1d EMA50
                if close[i] < lowest_low or close[i] < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price closes above Donchian high OR above 1d EMA50
                if close[i] > highest_high or close[i] > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals