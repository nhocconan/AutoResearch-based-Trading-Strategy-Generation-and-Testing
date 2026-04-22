#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12-hour Donchian channel breakout with 1-day trend filter and volume confirmation
    # Donchian(20) breakout captures momentum in trending markets
    # 1-day EMA34 filters trend direction: only take longs above EMA34, shorts below EMA34
    # Volume spike confirms institutional participation
    # Targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Donchian calculations
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian upper and lower bands (20-period)
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d volume MA for spike filter (20-period)
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Volume spike: current 12h volume > 2x 1d volume MA
    vol_spike = volume > 2.0 * vol_ma20_1d_aligned
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high + above 1d EMA34 (uptrend) + volume spike
            if close[i] > donch_high_aligned[i] and close[i] > ema34_1d_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + below 1d EMA34 (downtrend) + volume spike
            elif close[i] < donch_low_aligned[i] and close[i] < ema34_1d_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to middle of Donchian channel or trend reverses
            if position == 1:
                if close[i] < (donch_high_aligned[i] + donch_low_aligned[i]) / 2 or close[i] < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > (donch_high_aligned[i] + donch_low_aligned[i]) / 2 or close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_1dEMA34_Volume_Session_v1"
timeframe = "12h"
leverage = 1.0