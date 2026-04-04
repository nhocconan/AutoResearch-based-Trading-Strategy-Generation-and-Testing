#!/usr/bin/env python3
"""
exp_6454_1h_donchian20_4h_1d_ema_vol_v1
Hypothesis: 1h Donchian(20) breakout with 4h EMA(50) trend filter and 1d EMA(200) regime filter,
plus volume spike confirmation and session filter (08-20 UTC). Uses discrete position sizing
(0.20) to minimize fee churn. Designed for low trade frequency (target: 60-150 trades over 4 years)
to avoid fee drag while capturing medium-term trends in both bull and bear markets.
"""
name = "exp_6454_1h_donchian20_4h_1d_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    signals = np.zeros(n)
    
    # Pre-calculate session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    # 4h EMA(50) for trend filter
    close_4h = pd.Series(df_4h['close'])
    ema_4h = close_4h.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d EMA(200) for regime filter (bull/bear)
    close_1d = pd.Series(df_1d['close'])
    ema_1d_200 = close_1d.ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_1d_200_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_200)
    
    # 1h indicators for entry timing
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h Donchian(20) - highest high and lowest low of past 20 bars
    highest_20 = np.full(n, np.nan)
    lowest_20 = np.full(n, np.nan)
    for i in range(20, n):
        highest_20[i] = np.max(high[i-20:i])
        lowest_20[i] = np.min(low[i-20:i])
    
    # 1h volume SMA(20) for volume spike filter
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    # Warmup: need enough data for all indicators
    warmup = max(200, 20)  # 1d EMA200 needs 200 bars
    
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    
    for i in range(warmup, n):
        if not in_session[i]:
            signals[i] = 0.0
            if position_side != 0:
                signals[i] = 0.0  # close position outside session
                position_side = 0
            continue
        
        # Skip if any HTF data is not aligned yet
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_200_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Regime filter: price above/below 1d EMA200
        bull_regime = close[i] > ema_1d_200_aligned[i]
        bear_regime = close[i] < ema_1d_200_aligned[i]
        
        # Trend filter: 4h EMA50 direction
        uptrend = ema_4h_aligned[i] > ema_4h_aligned[i-1] if i > 0 else False
        downtrend = ema_4h_aligned[i] < ema_4h_aligned[i-1] if i > 0 else False
        
        # Donchian breakout conditions
        breakout_up = i >= 20 and not np.isnan(highest_20[i]) and close[i] > highest_20[i]
        breakout_down = i >= 20 and not np.isnan(lowest_20[i]) and close[i] < lowest_20[i]
        
        # Volume confirmation: volume > 1.5 * 20-period SMA
        vol_spike = i >= 20 and not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]
        
        # Entry logic
        if position_side == 0:  # flat, look for entry
            # Long: bull regime + uptrend + breakout up + volume spike
            if bull_regime and uptrend and breakout_up and vol_spike:
                signals[i] = 0.20
                position_side = 1
                entry_price = close[i]
            # Short: bear regime + downtrend + breakout down + volume spike
            elif bear_regime and downtrend and breakout_down and vol_spike:
                signals[i] = -0.20
                position_side = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position_side == 1:  # long position
            # Exit conditions: breakdown or stoploss
            lowest_10 = np.min(low[i-10:i]) if i >= 10 else low[i]
            stoploss_level = entry_price - 2.5 * (entry_price - lowest_10)  # approximate 2.5*ATR
            
            if close[i] < lowest_10 or close[i] < stoploss_level:
                signals[i] = 0.0  # exit long
                position_side = 0
            else:
                signals[i] = 0.20  # maintain long
        elif position_side == -1:  # short position
            # Exit conditions: breakout or stoploss
            highest_10 = np.max(high[i-10:i]) if i >= 10 else high[i]
            stoploss_level = entry_price + 2.5 * (highest_10 - entry_price)  # approximate 2.5*ATR
            
            if close[i] > highest_10 or close[i] > stoploss_level:
                signals[i] = 0.0  # exit short
                position_side = 0
            else:
                signals[i] = -0.20  # maintain short
    
    return signals