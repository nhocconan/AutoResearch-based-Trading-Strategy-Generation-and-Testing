#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian breakout with 1d EMA200 trend filter and volume confirmation
    # Donchian channels provide clear support/resistance levels. Breakouts with volume
    # confirm institutional participation. 1d EMA200 ensures alignment with daily trend.
    # This combination reduces false breakouts and improves win rate in both bull and bear markets.
    # Focus on 4h timeframe with strict entry conditions to limit trades to 20-50/year.
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian Channels (20-period)
    donchian_period = 20
    upper_donchian = pd.Series(high_4h).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_donchian = pd.Series(low_4h).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Align Donchian Channels to 4h timeframe (already aligned but keeping for clarity)
    upper_donchian_aligned = align_htf_to_ltf(prices, df_4h, upper_donchian)
    lower_donchian_aligned = align_htf_to_ltf(prices, df_4h, lower_donchian)
    
    # Load 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if data not ready or outside session
        if (np.isnan(upper_donchian_aligned[i]) or np.isnan(lower_donchian_aligned[i]) or
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above upper Donchian with volume + price above 1d EMA200 (uptrend)
            if close[i] > upper_donchian_aligned[i] and vol_spike[i] and close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below lower Donchian with volume + price below 1d EMA200 (downtrend)
            elif close[i] < lower_donchian_aligned[i] and vol_spike[i] and close[i] < ema200_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian level or trend reversal vs 1d EMA200
            if position == 1:
                if close[i] < lower_donchian_aligned[i] or close[i] < ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > upper_donchian_aligned[i] or close[i] > ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_1dEMA200_Volume_Session_v1"
timeframe = "4h"
leverage = 1.0