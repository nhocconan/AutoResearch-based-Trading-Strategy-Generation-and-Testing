#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12-hour Donchian breakout with 1-week EMA200 trend filter and volume spike confirmation
    # Works in bull/bear via trend filter: only take long in strong uptrend, short in strong downtrend.
    # Donchian breakouts capture momentum; EMA200 filters trend strength; volume confirms breakout validity.
    # Targets ~15-25 trades/year to minimize fee drag and avoid overtrading.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Donchian channel calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channel (20-period) on 12h timeframe
    # Using previous period to avoid look-ahead
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    
    # Calculate rolling max/min using pandas for proper handling
    high_series = pd.Series(prev_high_12h)
    low_series = pd.Series(prev_low_12h)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (already aligned via Donchian calculation)
    # No need to align as we're using 12h data directly for 12h timeframe strategy
    
    # Load 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Volume spike filter (20-period on 12h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):  # Start after warmup for EMA200
        # Skip if data not ready or outside session
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema200_1w_aligned[i]) or np.isnan(vol_ma20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume + price above 1w EMA200 (strong uptrend)
            if close[i] > donchian_high[i] and vol_spike[i] and close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volume + price below 1w EMA200 (strong downtrend)
            elif close[i] < donchian_low[i] and vol_spike[i] and close[i] < ema200_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian level or trend reversal vs 1w EMA200
            if position == 1:
                if close[i] < donchian_low[i] or close[i] < ema200_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donchian_high[i] or close[i] > ema200_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian_20_Breakout_1wEMA200_Volume_Session_v1"
timeframe = "12h"
leverage = 1.0