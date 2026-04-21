#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA(50) trend filter and volume spike confirmation.
# Long when price breaks above upper Donchian in uptrend (12h EMA50 rising), short when breaks below lower Donchian in downtrend.
# Volume > 2x 20-period average confirms breakout strength. Uses EMA slope to filter weak trends and avoid chop.
# Target: 25-40 trades/year by requiring strong trend + volume + breakout alignment.
# Works in bull/bear: EMA slope filter ensures only strong trends are traded, avoiding whipsaws in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate EMA slope (5-period change) to determine trend direction
    ema_slope = np.zeros_like(ema_50)
    ema_slope[5:] = (ema_50[5:] - ema_50[:-5]) / 5  # 5-period change
    
    # Align EMA and slope to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    ema_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_slope)
    
    # Calculate 20-period Donchian channels on 4h data
    high_roll = prices['high'].rolling(window=20, min_periods=20).max()
    low_roll = prices['low'].rolling(window=20, min_periods=20).min()
    upper = high_roll.values
    lower = low_roll.values
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(ema_50_aligned[i]) or np.isnan(ema_slope_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(upper[i]) or np.isnan(lower[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 2x 20-period average
        volume_confirm = volume > 2.0 * vol_ma[i]
        
        # Trend filter: strong uptrend (EMA50 rising) or strong downtrend (EMA50 falling)
        strong_uptrend = ema_slope_aligned[i] > 0
        strong_downtrend = ema_slope_aligned[i] < 0
        
        if position == 0:
            if volume_confirm:
                # Long: price breaks above upper Donchian in uptrend
                if price > upper[i] and strong_uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below lower Donchian in downtrend
                elif price < lower[i] and strong_downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price breaks below lower Donchian (failed breakout) or trend turns down
                if price < lower[i] or ema_slope_aligned[i] < 0:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price breaks above upper Donchian (failed breakdown) or trend turns up
                if price > upper[i] or ema_slope_aligned[i] > 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0