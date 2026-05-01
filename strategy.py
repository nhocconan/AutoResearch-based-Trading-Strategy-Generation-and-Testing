#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and ATR-based volatility filter.
# Uses 4h Donchian channel breakouts for trend continuation, filtered by 12h EMA50 trend direction.
# ATR-based volatility filter ensures trades only occur during sufficient volatility regimes.
# Works in both bull (buy upper band with uptrend) and bear (sell lower band with downtrend).
# Discrete position sizing (0.25) balances return and drawdown. Target: 75-200 trades over 4 years.

name = "4h_Donchian20_Breakout_12hEMA50_ATRFilter_v1"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR(14) for volatility filter on 12h timeframe
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    atr_14_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    
    # Calculate 4h Donchian channel (20-period)
    donchian_period = 20
    upper_channel = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, donchian_period) + 1  # 51
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(atr_14_12h_aligned[i]) or
            np.isnan(upper_channel[i]) or
            np.isnan(lower_channel[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: 12h EMA50 direction
        uptrend = curr_close > ema_50_12h_aligned[i]
        downtrend = curr_close < ema_50_12h_aligned[i]
        
        # Volatility filter: ATR > 0.5% of price (ensures sufficient volatility)
        volatility_filter = atr_14_12h_aligned[i] > (curr_close * 0.005)
        
        # Donchian breakout conditions
        breakout_up = curr_close > upper_channel[i-1]  # Break above previous upper band
        breakout_down = curr_close < lower_channel[i-1]  # Break below previous lower band
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above upper channel AND uptrend AND volatility filter
            if breakout_up and uptrend and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below lower channel AND downtrend AND volatility filter
            elif breakout_down and downtrend and volatility_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on breakdown below lower channel (reversal signal)
            if curr_close < lower_channel[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on breakout above upper channel (reversal signal)
            if curr_close > upper_channel[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals