#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and ATR-based stoploss.
- Primary timeframe: 1d for lower trade frequency (target: 30-100 total over 4 years).
- HTF: 1w EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Entry: Long when price breaks above 20-day high AND 1w EMA50 bullish.
         Short when price breaks below 20-day low AND 1w EMA50 bearish.
- Exit: ATR trailing stop (3*ATR) or reversal signal.
- Signal size: 0.25 discrete to minimize fee churn and control drawdown.
- Volume filter: Current day volume > 1.5 * 20-day volume MA to confirm participation.
This strategy captures medium-term trends filtered by weekly momentum, with volume
confirmation to avoid false breakouts. Works in both bull and bear markets by only
taking trades in the direction of the 1w trend. ATR stoploss manages risk during
volatile periods. Designed for low trade frequency to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    df_1w_close = df_1w['close'].values
    ema_1w = pd.Series(df_1w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-day Donchian channels (using 1d data implicitly via daily bars)
    # Since we're on 1d timeframe, we can calculate directly
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate ATR(14) for stoploss
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current day volume > 1.5 * 20-day volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 1d
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14, 20)  # EMA50, Donchian20, ATR14, VolMA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for breakout signals with volume spike
            if volume_spike[i]:
                # Bullish breakout: price > 20-day high AND 1w EMA50 bullish (close > EMA)
                if curr_close > highest_high[i] and curr_close > ema_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                # Bearish breakout: price < 20-day low AND 1w EMA50 bearish (close < EMA)
                elif curr_close < lowest_low[i] and curr_close < ema_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    lowest_since_entry = curr_close
        elif position == 1:
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            
            # Check exit conditions: ATR trailing stop or reversal
            long_stop = highest_since_entry - (3.0 * atr[i])
            if curr_close <= long_stop or curr_close < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            
            # Check exit conditions: ATR trailing stop or reversal
            short_stop = lowest_since_entry + (3.0 * atr[i])
            if curr_close >= short_stop or curr_close > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Trend_ATRStop_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0