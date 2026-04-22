#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4-hour Donchian breakout with 1-day ATR filter and volume confirmation
    # Breakouts capture momentum in trending markets, ATR filter avoids choppy conditions,
    # volume confirms institutional participation. Works in bull/bear via breakout direction.
    # Targets ~30 trades/year to minimize fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Load 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4-hour Donchian channels (20-period)
    # Use rolling window with min_periods to avoid look-ahead
    high_ma20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_ma20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (already on 4h, but ensure proper alignment)
    # Since we calculated on 4h data, we need to align back to the original 4h resolution
    # The Donchian values are already aligned to 4h bars
    
    # Calculate 1-day ATR (14-period) for volatility filter
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Handle first bar
    tr[0] = tr1[0]
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align ATR to 4h timeframe
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    
    # Volume spike filter (20-period on 4h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(high_ma20[i]) or np.isnan(low_ma20[i]) or
            np.isnan(atr14_aligned[i]) or np.isnan(vol_ma20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian upper band with volume + ATR filter (low volatility)
            if close[i] > high_ma20[i] and vol_spike[i] and atr14_aligned[i] < np.mean(atr14_aligned[max(0, i-50):i+1]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower band with volume + ATR filter (low volatility)
            elif close[i] < low_ma20[i] and vol_spike[i] and atr14_aligned[i] < np.mean(atr14_aligned[max(0, i-50):i+1]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian band or volatility increases
            if position == 1:
                if close[i] < low_ma20[i] or atr14_aligned[i] > 1.5 * np.mean(atr14_aligned[max(0, i-50):i+1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > high_ma20[i] or atr14_aligned[i] > 1.5 * np.mean(atr14_aligned[max(0, i-50):i+1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian_20_Breakout_ATRFilter_Volume_Session_v1"
timeframe = "4h"
leverage = 1.0