#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation
# Donchian channels provide robust structure for breakouts in both bull and bear markets
# 1d EMA50 ensures we only trade breakouts in the direction of the higher timeframe trend
# Volume spike (2.0x 20-period average) confirms institutional participation and reduces false breakouts
# ATR-based stoploss (2.0x ATR) manages risk during adverse moves
# Discrete sizing 0.25 minimizes fee churn. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_1dEMA50_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    start_idx = max(100, 50, 20, 14)  # warmup for EMA50, Donchian, volume MA, ATR
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        curr_atr = atr[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: break above Donchian upper AND above 1d EMA50 (uptrend)
                if curr_high > curr_highest_high and curr_close > curr_ema_50_1d:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    stop_price = curr_close - 2.0 * curr_atr
                # Bearish entry: break below Donchian lower AND below 1d EMA50 (downtrend)
                elif curr_low < curr_lowest_low and curr_close < curr_ema_50_1d:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    stop_price = curr_close + 2.0 * curr_atr
        
        elif position == 1:  # Long position
            # Exit conditions: price drops below Donchian lower OR stoploss hit
            if curr_close < curr_lowest_low or curr_close <= stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Trail stoploss up: move stop to breakeven + 0.5*ATR after 1*ATR profit
                if curr_close >= entry_price + 1.0 * curr_atr:
                    stop_price = max(stop_price, entry_price + 0.5 * curr_atr)
        
        elif position == -1:  # Short position
            # Exit conditions: price rises above Donchian upper OR stoploss hit
            if curr_close > curr_highest_high or curr_close >= stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Trail stoploss down: move stop to breakeven - 0.5*ATR after 1*ATR profit
                if curr_close <= entry_price - 1.0 * curr_atr:
                    stop_price = min(stop_price, entry_price - 0.5 * curr_atr)
    
    return signals