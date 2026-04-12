#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout + volume confirmation + chop regime filter
    # Primary timeframe: 12h for lower trade frequency and reduced fee drag
    # HTF: 1d for trend direction and regime detection
    # Donchian breakouts capture momentum; volume confirms institutional participation
    # Chop regime filter avoids whipsaw in ranging markets
    # Works in bull/bear by aligning with higher timeframe trend via price action
    # Target: 12-37 trades/year per symbol (50-150 total over 4 years)
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and ATR (chop filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d ATR(14) for volatility measurement
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
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high_1d = np.full(len(df_1d), np.nan)
    donchian_low_1d = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        donchian_high_1d[i] = np.max(high_1d[i-19:i+1])
        donchian_low_1d[i] = np.min(low_1d[i-19:i+1])
    
    donchian_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
    # Calculate 1d ATR(14) moving average (50-period) for chop regime
    atr_ma_50 = np.full(len(df_1d), np.nan)
    for i in range(49, len(df_1d)):
        if i == 49:
            atr_ma_50[i] = np.mean(atr14_1d[i-49:i+1])
        else:
            atr_ma_50[i] = (atr_ma_50[i-1] * 49 + atr14_1d[i]) / 50
    
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    
    # Calculate 12h Donchian channels (20-period) for entry signals
    donchian_high_12h = np.full(n, np.nan)
    donchian_low_12h = np.full(n, np.nan)
    for i in range(19, n):
        donchian_high_12h[i] = np.max(high[i-19:i+1])
        donchian_low_12h[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(donchian_high_1d_aligned[i]) or np.isnan(donchian_low_1d_aligned[i]) or
            np.isnan(atr14_1d_aligned[i]) or np.isnan(atr_ma_50_aligned[i]) or
            np.isnan(donchian_high_12h[i]) or np.isnan(donchian_low_12h[i])):
            signals[i] = 0.0
            continue
        
        # Chop regime filter: avoid extreme volatility conditions
        # Chop = high volatility (panic) or low volatility (ranging)
        atr_ratio = atr14_1d_aligned[i] / atr_ma_50_aligned[i]
        # Trade only when volatility is between 0.6x and 1.8x of 50-period average
        if atr_ratio < 0.6 or atr_ratio > 1.8:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = np.mean(volume[:i+1]) if i > 0 else volume[i]
        
        if volume[i] < 1.5 * vol_ma_20:
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above 12h Donchian high + HTF trend alignment
        long_breakout = close[i] > donchian_high_12h[i]
        # HTF trend filter: price above 1d Donchian middle (bullish bias)
        donchian_middle_1d = (donchian_high_1d_aligned[i] + donchian_low_1d_aligned[i]) / 2
        htf_bullish = close[i] > donchian_middle_1d
        
        # Short entry: price breaks below 12h Donchian low + HTF trend alignment
        short_breakout = close[i] < donchian_low_12h[i]
        # HTF trend filter: price below 1d Donchian middle (bearish bias)
        htf_bearish = close[i] < donchian_middle_1d
        
        if long_breakout and htf_bullish and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and htf_bearish and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and (close[i] < donchian_low_12h[i] or not htf_bullish):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > donchian_high_12h[i] or not htf_bearish):
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

name = "12h_1d_donchian_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0