#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1d EMA50 Trend + Volume Spike
Hypothesis: Donchian channel breakouts capture strong momentum. Aligned with 1d EMA50 trend and volume confirmation,
this strategy targets significant moves while avoiding choppy markets. Designed for 12h timeframe to achieve 12-37 trades/year.
Works in bull via long breakouts above upper band with uptrend, and in bear via short breakouts below lower band with downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Load 1d data for Donchian channels (using 1d high/low)
    # Donchian(20): upper = max(high of last 20 days), lower = min(low of last 20 days)
    df_1d_for_donch = get_htf_data(prices, '1d')  # Can reload as it's cached
    high_1d = df_1d_for_donch['high'].values
    low_1d = df_1d_for_donch['low'].values
    
    # Calculate Donchian channels on 1d
    donch_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h (no extra delay as based on completed 1d bar)
    donch_upper_aligned = align_htf_to_ltf(prices, df_1d_for_donch, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_1d_for_donch, donch_lower)
    
    # Volume confirmation: current volume > 2.0 * 30-period average (stricter for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(30, 50)  # volume MA, EMA50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_upper_aligned[i]) or np.isnan(donch_lower_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA50
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require ALL conditions: Donchian breakout + trend + volume
            # Long: price breaks above Donchian upper AND bullish bias AND volume spike
            long_entry = (curr_high > donch_upper_aligned[i]) and bullish_bias and vol_spike
            # Short: price breaks below Donchian lower AND bearish bias AND volume spike
            short_entry = (curr_low < donch_lower_aligned[i]) and bearish_bias and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below Donchian lower (mean reversion) OR loss of bullish bias
            if (curr_low < donch_lower_aligned[i]) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Donchian upper (mean reversion) OR loss of bearish bias
            if (curr_high > donch_upper_aligned[i]) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0