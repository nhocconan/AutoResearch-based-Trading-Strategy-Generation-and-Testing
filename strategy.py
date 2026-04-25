#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + 1w EMA50 Trend + Volume Spike
Hypothesis: Daily Donchian channel breakouts capture significant momentum moves.
Alignment with weekly EMA50 ensures trend-following in both bull and bear markets.
Volume confirmation filters false breakouts. Discrete sizing (0.25) targets ~30-100 trades over 4 years.
Works in bull/bear by adapting to weekly EMA50 direction - long only when price > EMA50,
short only when price < EMA50.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA50 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian (20) + weekly EMA (50)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_50_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_50_aligned[i]
        
        # Donchian channel (20-period) - using completed daily bars only
        lookback_start = max(0, i-19)
        lookback_end = i  # current bar is forming, use previous 20 completed bars
        if lookback_end - lookback_start >= 20:
            donchian_high = np.max(high[lookback_start:lookback_end])
            donchian_low = np.min(low[lookback_start:lookback_end])
        else:
            donchian_high = np.max(high[:i]) if i > 0 else curr_high
            donchian_low = np.min(low[:i]) if i > 0 else curr_low
        
        # Volume spike: current volume > 2.0 * 20-day average
        vol_lookback_start = max(0, i-19)
        vol_ma_20 = np.mean(volume[vol_lookback_start:i+1]) if i >= 19 else np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout conditions
        bullish_breakout = curr_close > donchian_high and volume_spike
        bearish_breakout = curr_close < donchian_low and volume_spike
        
        # Exit conditions: reverse Donchian breakout or trend rejection
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Exit on bearish breakout or price below weekly EMA50
                if bearish_breakout or curr_close < ema_trend:
                    exit_signal = True
            elif position == -1:
                # Exit on bullish breakout or price above weekly EMA50
                if bullish_breakout or curr_close > ema_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Donchian breakout + trend alignment + volume
        if position == 0:
            # Long: bullish breakout AND price above weekly EMA50
            long_condition = bullish_breakout and (curr_close > ema_trend)
            # Short: bearish breakout AND price below weekly EMA50
            short_condition = bearish_breakout and (curr_close < ema_trend)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0