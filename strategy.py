#!/usr/bin/env python3
"""
4h_adaptive_trend_breakout_v1
Hypothesis: On 4-hour timeframe, use adaptive trend detection (EMA20 vs EMA50) combined with Donchian breakout and volume confirmation. 
Trend-following in strong trends (ADX>25) and mean-reversion in choppy markets (ADX<20) with breakout entries. 
Designed for low frequency (20-50 trades/year) to minimize fee flood.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_adaptive_trend_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ADX calculation for regime filter
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    
    # Smoothed values
    tr_ma = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    dm_plus_ma = pd.Series(dm_plus).ewm(span=14, adjust=False).mean().values
    dm_minus_ma = pd.Series(dm_minus).ewm(span=14, adjust=False).mean().values
    
    # DI and ADX
    di_plus = 100 * dm_plus_ma / tr_ma
    di_minus = 100 * dm_minus_ma / tr_ma
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # EMAs for trend
    ema20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
    ema50 = pd.Series(close).ewm(span=50, adjust=False).mean().values
    
    # Donchian channels
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    donch_high = rolling_max(high, 20)
    donch_low = rolling_min(low, 20)
    
    # Volume average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(adx[i]) or np.isnan(ema20[i]) or np.isnan(ema50[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_avg[i]):
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Regime: ADX > 25 = trending, ADX < 20 = choppy
        is_trending = adx[i] > 25
        is_choppy = adx[i] < 20
        
        if position == 1:  # Long position
            # Exit: trend reversal or opposite touch
            if is_trending and ema20[i] < ema50[i]:  # trend reversal
                position = 0
                signals[i] = 0.0
            elif not is_trending and close[i] <= donch_low[i]:  # mean reversion exit
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend reversal or opposite touch
            if is_trending and ema20[i] > ema50[i]:  # trend reversal
                position = 0
                signals[i] = 0.0
            elif not is_trending and close[i] >= donch_high[i]:  # mean reversion exit
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if is_trending:
                # Trend following: breakout in trend direction
                if ema20[i] > ema50[i] and close[i] > donch_high[i] and vol_confirm:  # uptrend breakout
                    position = 1
                    signals[i] = 0.25
                elif ema20[i] < ema50[i] and close[i] < donch_low[i] and vol_confirm:  # downtrend breakout
                    position = -1
                    signals[i] = -0.25
            else:
                # Choppy market: mean reversion at extremes
                if close[i] < donch_low[i] and vol_confirm:  # oversold bounce
                    position = 1
                    signals[i] = 0.25
                elif close[i] > donch_high[i] and vol_confirm:  # overbought pullback
                    position = -1
                    signals[i] = -0.25
    
    return signals