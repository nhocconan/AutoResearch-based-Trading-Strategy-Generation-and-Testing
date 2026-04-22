#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 12-hour Donchian breakout with 1-day trend filter and volume confirmation
    # Works in bull/bear via 1-day EMA100 trend filter: only take long in uptrend, short in downtrend.
    # Donchian breakouts capture momentum; EMA100 filters trend; volume confirms breakout strength.
    # Targets ~20-30 trades/year to minimize fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Load 1d data for EMA100 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # Calculate Donchian channels (20-period) on 12h data
    # Use rolling window on high/low
    donchian_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (already aligned via get_htf_data + align)
    donchian_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_12h)
    donchian_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_12h)
    
    # Volume spike filter (15-period on 12h)
    vol_ma15 = pd.Series(volume).rolling(window=15, min_periods=15).mean().values
    vol_spike = volume > 1.5 * vol_ma15  # Require 1.5x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(donchian_high_12h_aligned[i]) or np.isnan(donchian_low_12h_aligned[i]) or
            np.isnan(ema100_1d_aligned[i]) or np.isnan(vol_ma15[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume + price above 1d EMA100 (uptrend)
            if close[i] > donchian_high_12h_aligned[i] and vol_spike[i] and close[i] > ema100_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volume + price below 1d EMA100 (downtrend)
            elif close[i] < donchian_low_12h_aligned[i] and vol_spike[i] and close[i] < ema100_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian level or trend reversal vs 1d EMA100
            if position == 1:
                if close[i] < donchian_low_12h_aligned[i] or close[i] < ema100_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donchian_high_12h_aligned[i] or close[i] > ema100_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian_20_Breakout_1dEMA100_Volume_Session_v1"
timeframe = "12h"
leverage = 1.0