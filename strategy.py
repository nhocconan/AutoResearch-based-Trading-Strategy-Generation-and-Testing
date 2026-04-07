#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h ADX Trend + 4h/1d EMA Filter + Session Filter (08-20 UTC)
# Hypothesis: ADX > 25 filters trending markets, while 4h/1d EMA alignment ensures
# higher timeframe trend confirmation. Session filter reduces noise during low-
# liquidity hours. Uses ADX for trend strength (not direction) with price vs EMA
# for direction. Designed for low trade frequency (15-37/year) to avoid fee drag.

name = "1h_adx_trend_4h1d_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h and 1d data for trend filters (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    close_1d = df_1d['close'].values
    
    # 4h EMA(50) and 1d EMA(50) for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ADX(14) for trend strength
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # ADX > 25 indicates strong trend
        strong_trend = adx[i] > 25
        
        if position == 1:  # Long position
            # Exit: trend weakens OR price closes below 4h/1d EMA
            if not strong_trend or close[i] < ema_50_4h_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: trend weakens OR price closes above 4h/1d EMA
            if not strong_trend or close[i] > ema_50_4h_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            if strong_trend and in_session:
                # Enter long: price above both 4h and 1d EMA
                if close[i] > ema_50_4h_aligned[i] and close[i] > ema_50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Enter short: price below both 4h and 1d EMA
                elif close[i] < ema_50_4h_aligned[i] and close[i] < ema_50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals