#!/usr/bin/env python3
"""
Hypothesis: 1-day Donchian Breakout + 1-week Trend + Volume Confirmation.
In bull market (1-week close > 1-week EMA50): long when price breaks above Donchian upper (20).
In bear market (1-week close < 1-week EMA50): short when price breaks below Donchian lower (20).
Volume must be above 20-period average to confirm.
Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1W TREND FILTER (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    one_w_close = df_1w['close'].values
    one_w_ema = pd.Series(one_w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    one_w_ema_aligned = align_htf_to_ltf(prices, df_1w, one_w_ema)
    
    # === 1D DONCHIAN CHANNEL (20) ===
    donchian_length = 20
    donchian_upper = pd.Series(high).rolling(window=donchian_length, min_periods=donchian_length).max().values
    donchian_lower = pd.Series(low).rolling(window=donchian_length, min_periods=donchian_length).min().values
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        if np.isnan(one_w_ema_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1w EMA
        bull_trend = one_w_ema_aligned[i] > 0  # Always true since price > 0
        bull_trend = close[i] > one_w_ema_aligned[i]  # Price above EMA = bullish
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower OR trend turns bearish
            if close[i] < donchian_lower[i] or not bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper OR trend turns bullish
            if close[i] > donchian_upper[i] or bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with trend alignment
            if bull_trend:
                # In bull market: long on breakout above Donchian upper
                if high[i] > donchian_upper[i]:
                    position = 1
                    signals[i] = 0.25
            else:
                # In bear market: short on breakout below Donchian lower
                if low[i] < donchian_lower[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals