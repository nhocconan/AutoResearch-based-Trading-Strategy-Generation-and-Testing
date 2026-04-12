#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + 1w/1d regime filter
    # Elder Ray (Bull/Bear Power) measures trend strength via EMA13
    # Weekly trend filter: only take trades in direction of 1w EMA34
    # Daily volatility filter: avoid low volatility chop
    # Works in bull/bear by aligning with higher timeframe trend
    # Target: 12-37 trades/year per symbol.
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA13 (Elder Ray) and ATR14 (volatility filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA13 for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align 1d Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 1d ATR14 for volatility filter
    tr_1d = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if i == 0:
            tr_1d[i] = high_1d[i] - low_1d[i]
        else:
            tr_1d[i] = max(
                high_1d[i] - low_1d[i],
                abs(high_1d[i] - close_1d[i-1]),
                abs(low_1d[i] - close_1d[i-1])
            )
    
    atr14_1d = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        if i == 13:
            atr14_1d[i] = np.mean(tr_1d[i-13:i+1])
        else:
            atr14_1d[i] = (atr14_1d[i-1] * 13 + tr_1d[i]) / 14
    
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 for trend direction
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 6h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(atr14_1d_aligned[i]) or np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely low volatility (chop)
        # Also avoid extremely high volatility (panic)
        vol_ma_50 = np.full(len(df_1d), np.nan)
        for j in range(49, len(df_1d)):
            if j == 49:
                vol_ma_50[j] = np.mean(atr14_1d[j-49:j+1])
            else:
                vol_ma_50[j] = (vol_ma_50[j-1] * 49 + atr14_1d[j]) / 50
        vol_ma_50_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_50)
        
        if np.isnan(vol_ma_50_aligned[i]):
            signals[i] = 0.0
            continue
            
        vol_ratio = atr14_1d_aligned[i] / vol_ma_50_aligned[i]
        # Trade only when volatility is between 0.5x and 2.0x of 50-period average
        if vol_ratio < 0.5 or vol_ratio > 2.0:
            signals[i] = 0.0
            continue
        
        # Regime: only take trades in direction of weekly trend
        weekly_uptrend = close[i] > ema34_1w_aligned[i]
        weekly_downtrend = close[i] < ema34_1w_aligned[i]
        
        # Elder Ray signals
        # Long when Bull Power > 0 and increasing (strong bullish momentum)
        # Short when Bear Power < 0 and decreasing (strong bearish momentum)
        if i > 0:
            bull_power_prev = bull_power_aligned[i-1]
            bear_power_prev = bear_power_aligned[i-1]
        else:
            bull_power_prev = bull_power_aligned[i]
            bear_power_prev = bear_power_aligned[i]
        
        bull_power_rising = bull_power_aligned[i] > bull_power_prev
        bear_power_falling = bear_power_aligned[i] < bear_power_prev
        
        long_entry = (bull_power_aligned[i] > 0) and bull_power_rising and weekly_uptrend
        short_entry = (bear_power_aligned[i] < 0) and bear_power_falling and weekly_downtrend
        
        # Exit when power fades or reverses
        long_exit = (bull_power_aligned[i] <= 0) or not bull_power_rising
        short_exit = (bear_power_aligned[i] >= 0) or not bear_power_falling
        
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

name = "6h_1d_1w_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0