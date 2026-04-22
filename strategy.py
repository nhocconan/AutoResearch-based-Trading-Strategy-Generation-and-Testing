#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(15) breakout with weekly EMA34 trend filter and volume spike
    # Targets 15-30 trades/year per symbol to minimize fee drag.
    # Donchian breakouts capture momentum; weekly EMA34 filters long-term trend;
    # volume spike confirms institutional interest. Works in bull/bear via trend filter.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Donchian(15) calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels (15-period) for each 12h bar
    donchian_upper = pd.Series(high_12h).rolling(window=15, min_periods=15).max().values
    donchian_lower = pd.Series(low_12h).rolling(window=15, min_periods=15).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    
    # Load weekly data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike filter (15-period)
    vol_ma15 = pd.Series(volume).rolling(window=15, min_periods=15).mean().values
    vol_spike = volume > 2.0 * vol_ma15  # Require 2x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma15[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above Donchian upper with volume + price above weekly EMA34 (uptrend)
            if close[i] > donchian_upper_aligned[i] and vol_spike[i] and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below Donchian lower with volume + price below weekly EMA34 (downtrend)
            elif close[i] < donchian_lower_aligned[i] and vol_spike[i] and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian level or trend reversal vs weekly EMA34
            if position == 1:
                if close[i] < donchian_lower_aligned[i] or close[i] < ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donchian_upper_aligned[i] or close[i] > ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian_15_Breakout_1wEMA34_Volume_Session_v1"
timeframe = "12h"
leverage = 1.0