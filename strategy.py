#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ADX regime filter and volume confirmation
# In strong trends (ADX > 25): breakout continuation above/below 20-period Donchian channels
# In weak trends/ranging (ADX <= 25): no new entries, only exits on opposite channel touch
# Volume confirmation requires current volume > 1.5x 20-period average volume
# Designed for 6h timeframe to capture medium-term trends while avoiding whipsaws
# Discrete position sizing 0.25 to limit trades to ~12-37/year and reduce fee drag
# Works in bull/bear markets: trend filter avoids false breakouts in ranging markets

name = "6h_1d_donchian_breakout_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) for ADX
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # Calculate 1d +DM and -DM for ADX
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Calculate 1d smoothed +DM, -DM, and TR
    atr_period = 14
    plus_dm_smooth = wilders_smoothing(plus_dm, atr_period)
    minus_dm_smooth = wilders_smoothing(minus_dm, atr_period)
    tr_smooth = wilders_smoothing(tr, atr_period)
    
    # Calculate 1d +DI and -DI
    plus_di = np.where(tr_smooth > 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth > 0, 100 * minus_dm_smooth / tr_smooth, 0)
    
    # Calculate 1d DX and ADX
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_1d = wilders_smoothing(dx, atr_period)
    
    # Calculate 6h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute volume confirmation array
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * avg_volume
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: strong trend when ADX > 25
        strong_trend = adx_1d_aligned[i] > 25.0
        
        if position == 1:  # Long position
            if strong_trend:
                # Exit long if price breaks below Donchian low or trend weakens
                if close[i] < lowest_low[i] or not strong_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:
                # In weak trend, exit long on any opposite signal
                if close[i] < lowest_low[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            if strong_trend:
                # Exit short if price breaks above Donchian high or trend weakens
                if close[i] > highest_high[i] or not strong_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:
                # In weak trend, exit short on any opposite signal
                if close[i] > highest_high[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if strong_trend and volume_confirmed[i]:
                # Enter long on breakout above Donchian high with volume confirmation
                if close[i] > highest_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short on breakout below Donchian low with volume confirmation
                elif close[i] < lowest_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals