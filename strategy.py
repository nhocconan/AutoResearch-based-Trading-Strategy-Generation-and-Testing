#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (price > 1d EMA50 for long, < for short) and ATR-based volatility filter.
# Uses Donchian channel breakouts as the primary signal, filtered by daily trend and sufficient volatility (ATR > 0.5 * ATR_ma).
# Works in bull markets (buy upside breakouts with uptrend) and bear markets (sell downside breakouts with downtrend).
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe.
# Discrete position sizing (0.25) to minimize fee churn.

name = "4h_Donchian20_Breakout_1dEMA50_ATRFilter_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma = pd.Series(atr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Donchian channel (20-period) from previous candles
    # Using rolling window on past data only (no look-ahead)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for EMA50, ATR, and Donchian
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr[i]) or
            np.isnan(atr_ma[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        
        # Trend filter: 1d EMA50 direction
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Volatility filter: current ATR > 0.5 * 20-period ATR MA (ensures sufficient volatility)
        volatility_filter = atr[i] > (0.5 * atr_ma[i])
        
        # Donchian breakout conditions
        breakout_up = curr_close > highest_high[i]   # break above 20-period high
        breakout_down = curr_close < lowest_low[i]   # break below 20-period low
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout up AND uptrend AND volatility filter
            if breakout_up and uptrend and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: Breakout down AND downtrend AND volatility filter
            elif breakout_down and downtrend and volatility_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Donchian breakout down (reversal signal) or trend change
            if breakout_down or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Donchian breakout up (reversal signal) or trend change
            if breakout_up or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals