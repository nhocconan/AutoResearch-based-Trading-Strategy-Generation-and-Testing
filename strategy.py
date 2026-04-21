#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h trend filter and 1d volume confirmation.
# Long: price breaks above 4h Donchian high (20) AND 1h close > 1h VWAP AND 1d volume > 1.5x 20-day average
# Short: price breaks below 4h Donchian low (20) AND 1h close < 1h VWAP AND 1d volume > 1.5x 20-day average
# Uses 4h for trend direction (structure), 1h for entry timing precision, 1d for volume confirmation.
# Target: 15-35 trades/year by requiring multi-timeframe alignment + volume filter.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h for Donchian channels (trend structure)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Load 1d for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period high/low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume moving average (20-day)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 4h indicators to 1h
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Align 1d volume average to 1h
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1h VWAP (typical price * volume) / cumulative volume
    typical_price = (prices['high'] + prices['low'] + prices['close']) / 3
    vwap = (typical_price * prices['volume']).cumsum() / prices['volume'].cumsum()
    vwap = vwap.values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(vwap[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        close = prices['close'].iloc[i]
        vol_current = prices['volume'].iloc[i]
        
        # Breakout conditions
        breakout_up = close > donchian_high_aligned[i]
        breakout_down = close < donchian_low_aligned[i]
        
        # VWAP filter: price above/below VWAP for directional bias
        above_vwap = close > vwap[i]
        below_vwap = close < vwap[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirm = vol_current > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Enter long: upward breakout + above VWAP + volume confirmation
            if breakout_up and above_vwap and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Enter short: downward breakout + below VWAP + volume confirmation
            elif breakout_down and below_vwap and volume_confirm:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit conditions: opposite breakout OR volume drops below average
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian low OR volume drops
                if close < donchian_low_aligned[i]:
                    exit_signal = True
                elif vol_current < vol_ma_20_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above Donchian high OR volume drops
                if close > donchian_high_aligned[i]:
                    exit_signal = True
                elif vol_current < vol_ma_20_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Donchian_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0