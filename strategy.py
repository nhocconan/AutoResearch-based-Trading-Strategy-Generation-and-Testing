#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout + 1d trend + volume confirmation.
In bull market (1d close > 1d EMA100): long on Donchian(20) breakout above upper band.
In bear market (1d close < 1d EMA100): short on Donchian(20) breakout below lower band.
Volume must be above 20-period average to confirm breakout.
Target: 20-40 trades/year (80-160 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1D TREND FILTER (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    one_d_close = df_1d['close'].values
    one_d_ema = pd.Series(one_d_close).ewm(span=100, adjust=False, min_periods=100).mean().values
    one_d_ema_aligned = align_htf_to_ltf(prices, df_1d, one_d_ema)
    
    # === 4H DONCHIAN CHANNEL (LTF) ===
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === VOLUME CONFIRMATION (LTF) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(one_d_ema_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        bull_trend = one_d_close[-1] > one_d_ema[-1] if len(one_d_close) > 0 else False  # Use last known 1d value
        # Actually, use aligned value for current bar
        bull_trend = close[i] > one_d_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR trend turns bearish
            if close[i] < donchian_low[i] or not bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR trend turns bullish
            if close[i] > donchian_high[i] or bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry logic based on 1d trend and Donchian breakout
            if bull_trend:
                # In bull market: long on breakout above Donchian high
                if high[i] > donchian_high[i] and close[i] > donchian_high[i]:
                    position = 1
                    signals[i] = 0.25
            else:
                # In bear market: short on breakout below Donchian low
                if low[i] < donchian_low[i] and close[i] < donchian_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals