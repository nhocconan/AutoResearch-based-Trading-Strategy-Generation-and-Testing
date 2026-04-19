#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly 52-week high/low breakout with volume confirmation and 1w trend filter
# Uses tight entry conditions to limit trades (target: 20-50/year) and avoid fee drag
# Works in bull markets via 52-week high breakouts and in bear via 52-week low breakdowns
# Only trades when volume confirms breakout and higher timeframe trend aligns
name = "1d_Weekly52WeekHighBreakout_VolumeTrend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for multi-timeframe analysis (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1d 52-week high/low (250 periods)
    high_52w = pd.Series(high).rolling(window=250, min_periods=250).max().values
    low_52w = pd.Series(low).rolling(window=250, min_periods=250).min().values
    
    # 1d ATR for position sizing and stops
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 250
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1w_aligned[i]) or \
           np.isnan(high_52w[i]) or np.isnan(low_52w[i]) or np.isnan(atr_1d[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_1d[i]
        
        # Volume filter: current volume > 1.5x average volume (20-period)
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 1.5 * avg_volume
        
        if position == 0:
            # Long: breakout above 52-week high + volume + 1w uptrend
            if high[i] > high_52w[i-1] and volume_filter and price > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below 52-week low + volume + 1w downtrend
            elif low[i] < low_52w[i-1] and volume_filter and price < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below 52-week low or ATR-based stop
            if close[i] < low_52w[i] or close[i] < close[i-1] - 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above 52-week high or ATR-based stop
            if close[i] > high_52w[i] or close[i] > close[i-1] + 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals