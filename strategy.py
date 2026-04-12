#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian breakout with 1d EMA trend filter and chop regime
    # Donchian(20) breakout provides clear entry/exit levels
    # 1d EMA(50) ensures alignment with higher timeframe trend
    # Choppiness Index filter avoids whipsaw in ranging markets
    # Volume confirmation adds conviction to breakouts
    # Target: 20-40 trades/year per symbol to minimize fee drag
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA(50) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Choppiness Index on 1d (14-period)
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(14)
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.absolute(high_1d[1:] - close_1d[:-1]),
                       np.absolute(low_1d[1:] - close_1d[:-1]))
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align length
    
    atr_1d = np.full(len(tr_1d), np.nan)
    for i in range(14, len(tr_1d)):
        atr_1d[i] = np.nanmean(tr_1d[i-13:i+1])
    
    # Calculate CHOP(14)
    chop_1d = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        sum_atr = np.nansum(atr_1d[i-13:i+1])
        max_high = np.nanmax(high_1d[i-13:i+1])
        min_low = np.nanmin(low_1d[i-13:i+1])
        if max_high > min_low and sum_atr > 0:
            chop_1d[i] = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(14)
        else:
            chop_1d[i] = 50.0  # neutral when undefined
    
    # Align 1d indicators to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 1d EMA(50)
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        if i >= 20:
            vol_ma = np.mean(volume[max(0, i-20):i])
            vol_ratio = volume[i] / vol_ma if vol_ma > 0 else 1.0
        else:
            vol_ratio = 1.0
        
        # Chop regime: only trade when CHOP < 61.8 (avoid extremely choppy markets)
        chop_condition = chop_aligned[i] < 61.8
        
        # Donchian breakout signals with volume confirmation AND trend alignment
        breakout_long = (close[i] > highest_high[i]) and (vol_ratio > 1.5) and uptrend and chop_condition
        breakout_short = (close[i] < lowest_low[i]) and (vol_ratio > 1.5) and downtrend and chop_condition
        
        # Exit conditions: return to opposite Donchian level (middle)
        middle = (highest_high[i] + lowest_low[i]) / 2.0
        
        long_exit = close[i] < middle
        short_exit = close[i] > middle
        
        if breakout_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_short and position != -1:
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

name = "4h_1d_donchian_breakout_ema_chop_v1"
timeframe = "4h"
leverage = 1.0