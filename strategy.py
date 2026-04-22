#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Works in bull/bear by using 1d EMA34 for trend direction and volume spike for momentum confirmation.
# Entry: Price breaks Donchian high/low + volume > 1.5x average + price on correct side of 1d EMA34.
# Exit: Opposite Donchian break or trend reversal. Target: 20-50 trades/year (80-200 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Donchian(20) channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian high: max(high_4h, lookback=20)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian low: min(low_4h, lookback=20)
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 4h timeframe (already aligned by get_htf_data, but we need to shift for breakout)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate 4h volume average (20-period)
    vol_4h = df_4h['volume'].values
    vol_avg_20 = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian lookback period
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_avg_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high + uptrend + volume confirmation
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + downtrend + volume confirmation
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # Long position
                # Exit: Price breaks below Donchian low OR trend reverses
                if (close[i] < donchian_low_aligned[i] or 
                    close[i] < ema_34_1d_aligned[i]):
                    exit_signal = True
            else:  # position == -1, Short position
                # Exit: Price breaks above Donchian high OR trend reverses
                if (close[i] > donchian_high_aligned[i] or 
                    close[i] > ema_34_1d_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0