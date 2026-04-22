#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: Daily Donchian(20) breakout with 1-week EMA50 trend filter and volume confirmation
    # Donchian channels provide robust breakout signals in trending markets
    # Weekly EMA50 filters for primary trend direction (avoid counter-trend trades)
    # Volume confirmation ensures breakouts have institutional participation
    # Works in both bull (breakouts up) and bear (breakouts down) markets
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.2 * vol_ma20  # Require 1.2x average volume
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after indicators warm up
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high + above weekly EMA50 + volume confirmation
            if close[i] > donchian_high[i] and close[i] > ema50_1w_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + below weekly EMA50 + volume confirmation
            elif close[i] < donchian_low[i] and close[i] < ema50_1w_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian band or trend reversal
            if position == 1:
                if close[i] < donchian_low[i] or close[i] < ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donchian_high[i] or close[i] > ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian_Breakout_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0