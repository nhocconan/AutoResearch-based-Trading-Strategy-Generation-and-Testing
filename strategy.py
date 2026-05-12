# [Experiment 158398] 1d Donchian Breakout + Volume + Chop Filter
# Hypothesis: Breakouts from daily Donchian channels (20-day high/low) with volume confirmation
# and choppy market filter work in both bull and bear markets by capturing momentum bursts
# while avoiding false breakouts in ranging conditions. Designed for low frequency (10-25 trades/year)
# to minimize fee drag. Uses 1-week trend filter for higher timeframe context.

name = "1d_Donchian_Breakout_Volume_Chop"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 20-day Donchian channels
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i >= lookback - 1:
            donchian_high[i] = np.max(high[i - lookback + 1:i + 1])
            donchian_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(n):
        if i >= lookback - 1:
            vol_ma[i] = np.mean(volume[i - lookback + 1:i + 1])
    
    # Calculate Choppiness Index (14-period) for regime filter
    chop = np.full(n, np.nan)
    chop_period = 14
    for i in range(n):
        if i >= chop_period - 1:
            # True range
            tr1 = high[i - chop_period + 1:i + 1] - low[i - chop_period + 1:i + 1]
            tr2 = np.abs(high[i - chop_period + 1:i + 1] - np.roll(close[i - chop_period + 1:i + 1], 1))
            tr3 = np.abs(low[i - chop_period + 1:i + 1] - np.roll(close[i - chop_period + 1:i + 1], 1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = np.mean(tr)
            
            max_high = np.max(high[i - chop_period + 1:i + 1])
            min_low = np.min(low[i - chop_period + 1:i + 1])
            
            if max_high != min_low:
                chop[i] = 100 * np.log10(atr * chop_period / (max_high - min_low)) / np.log10(chop_period)
            else:
                chop[i] = 50  # neutral when no range
    
    # Weekly EMA34 for trend filter
    ema34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema_series = pd.Series(close_1w).ewm(span=34, adjust=False).mean()
        ema34_1w = ema_series.values
    
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, chop_period)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        # Choppy market filter: chop > 61.8 indicates ranging market (avoid breakouts)
        # chop < 38.2 indicates trending market (favor breakouts)
        chop_filter = chop[i] < 38.2  # Only trade in trending conditions
        
        # Weekly trend filter: price above/below weekly EMA34
        weekly_uptrend = close[i] > ema34_1w_aligned[i]
        weekly_downtrend = close[i] < ema34_1w_aligned[i]
        
        if position == 0:
            # LONG: Price breaks above Donchian high + volume spike + chop filter + weekly uptrend
            if (close[i] > donchian_high[i] and volume_spike and 
                chop_filter and weekly_uptrend):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + volume spike + chop filter + weekly downtrend
            elif (close[i] < donchian_low[i] and volume_spike and 
                  chop_filter and weekly_downtrend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low OR chop becomes too high (ranging)
            if close[i] < donchian_low[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high OR chop becomes too high (ranging)
            if close[i] > donchian_high[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals