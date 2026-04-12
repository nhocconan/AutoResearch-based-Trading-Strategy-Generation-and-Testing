#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + 12h trend filter
    # Elder Ray measures bull/bear power relative to EMA13
    # 12h EMA34 trend filter ensures we trade with higher timeframe momentum
    # Volume confirmation reduces false signals
    # Works in bull/bear by adapting to trend direction
    # Target: 50-150 total trades over 4 years (~12-37/year)
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 6h EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 6h Elder Ray components
    bull_power = high - ema13  # Bull power: high minus EMA
    bear_power = low - ema13   # Bear power: low minus EMA
    
    # Calculate 12h EMA34 for trend filter
    close_12h_s = pd.Series(close_12h)
    ema34_12h = close_12h_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate 6h volume spike filter (current volume > 1.5 * 20-period average)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    volume_spike = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 12h EMA34 direction
        uptrend = close_12h_s.iloc[-1] > ema34_12h_aligned[i] if len(close_12h_s) > 0 else False
        # Simplified: use current 12h price vs EMA
        # Find corresponding 12h bar index
        idx_12h = i // 2  # 2x 6h bars per 12h bar (approximate, but aligned array handles timing)
        if idx_12h < len(ema34_12h_aligned):
            uptrend = close[i] > ema34_12h_aligned[i]  # Use aligned 6h price vs 12h EMA
        else:
            uptrend = True  # Default to long if insufficient data
        
        # Elder Ray logic with volume confirmation
        if uptrend:
            # In uptrend: look for bullish signals
            long_entry = bull_power[i] > 0 and volume_spike[i] and bear_power[i] < 0
            short_entry = False  # Only short in downtrend
            long_exit = bull_power[i] < 0  # Exit when bull power turns negative
            short_exit = False
        else:
            # In downtrend: look for bearish signals
            long_entry = False  # Only long in uptrend
            short_entry = bear_power[i] < 0 and volume_spike[i] and bull_power[i] > 0
            long_exit = False
            short_exit = bear_power[i] > 0  # Exit when bear power turns positive
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_elder_ray_trend_v1"
timeframe = "6h"
leverage = 1.0