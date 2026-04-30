#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and ATR-based volatility filter
# Donchian channels provide clear breakout/breakdown levels based on 20-period high/low
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend whipsaws
# ATR(14) > 1.5x ATR(50) filter ensures sufficient volatility for meaningful moves
# Discrete sizing 0.25 minimizes fee churn. Target: 75-200 total trades over 4 years (19-50/year).
# Works in bull markets via upside breakouts and bear markets via downside breakdowns with trend filter.

name = "4h_Donchian20_1dEMA34_ATR_VolFilter_v2"
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
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) and ATR(50) for volatility filter
    high_low = high - low
    high_close_prev = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    low_close_prev = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(true_range).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr_14 > (1.5 * atr_50)
    
    # Calculate Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20)  # warmup for EMA and ATR
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_14[i]) or np.isnan(atr_50[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_vol_filter = vol_filter[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volatility filter and price above/below Donchian channels for breakout
            if curr_vol_filter:
                # Bullish entry: break above Donchian high with price > EMA34_1d
                if curr_close > curr_donchian_high and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: break below Donchian low with price < EMA34_1d
                elif curr_close < curr_donchian_low and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below Donchian low (breakdown) OR price crosses below EMA34_1d
            if curr_close < curr_donchian_low or curr_close < curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above Donchian high (breakout) OR price crosses above EMA34_1d
            if curr_close > curr_donchian_high or curr_close > curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals