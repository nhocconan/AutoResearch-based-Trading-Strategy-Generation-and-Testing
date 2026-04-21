#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Daily Pivot Point (PP) breakout with 1w EMA(20) trend filter and volume confirmation.
# Long when price breaks above R1 in uptrend (weekly EMA > previous week EMA), short when breaks below S1 in downtrend.
# Volume > 1.5x 20-period average confirms breakout. Uses weekly trend to avoid counter-trend trades.
# Pivot points calculated from prior day's OHLC. Target: 10-25 trades/year by requiring strong weekly trend + volume + pivot breakout.
# Works in bull/bear: weekly trend filter ensures alignment with higher timeframe momentum, reducing whipsaws in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Rising EMA = current > previous
    ema_rising = np.zeros_like(ema_20, dtype=bool)
    ema_rising[1:] = ema_20[1:] > ema_20[:-1]
    # Align to 1d timeframe
    ema_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_rising)
    
    # Calculate daily pivot points (using prior day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot Point (PP) = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Resistance 1 (R1) = (2 * PP) - L
    r1 = (2 * pp) - low_1d
    # Support 1 (S1) = (2 * PP) - H
    s1 = (2 * pp) - high_1d
    
    # Align pivot levels to 1d timeframe (already aligned as calculated from 1d data)
    # But we need to shift by 1 to use prior day's levels for today's breakout
    r1_shifted = np.roll(r1, 1)
    s1_shifted = np.roll(s1, 1)
    r1_shifted[0] = np.nan  # First day has no prior
    s1_shifted[0] = np.nan
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(ema_rising_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(r1_shifted[i]) or np.isnan(s1_shifted[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Trend filter: weekly EMA rising (uptrend)
        uptrend = ema_rising_aligned[i]
        downtrend = not uptrend  # Simple definition for clarity
        
        if position == 0:
            if volume_confirm:
                # Long: price breaks above R1 in uptrend
                if uptrend and price > r1_shifted[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below S1 in downtrend
                elif downtrend and price < s1_shifted[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price breaks below S1 (failed breakout) or trend turns down
                if price < s1_shifted[i] or downtrend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price breaks above R1 (failed breakdown) or trend turns up
                if price > r1_shifted[i] or uptrend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_PivotPoint_R1S1_Breakout_1wEMA20_Trend_Volume"
timeframe = "1d"
leverage = 1.0