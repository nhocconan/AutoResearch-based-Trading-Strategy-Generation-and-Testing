#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter (EMA50 slope) and volume confirmation.
# Long when price breaks above upper Donchian in uptrend (12h EMA50 rising), short when breaks below lower Donchian in downtrend.
# Volume > 1.5x 20-period average confirms breakout strength. EMA slope filters weak trends and avoids chop.
# Target: 20-50 trades/year by requiring strong trend + volume + breakout alignment.
# Works in bull/bear: EMA slope ensures only trending markets are traded, avoiding whipsaws in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    
    # Align EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate EMA50 slope (rising/falling) - trend direction
    ema_slope = np.diff(ema_50_aligned, prepend=ema_50_aligned[0])
    ema_rising = ema_slope > 0
    ema_falling = ema_slope < 0
    
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
        if np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(upper[i]) or np.isnan(lower[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Trend filter: EMA50 rising/falling
        trend_up = ema_rising[i]
        trend_down = ema_falling[i]
        
        if position == 0:
            if volume_confirm:
                # Long: price breaks above upper Donchian + EMA50 rising
                if price > upper[i] and trend_up:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below lower Donchian + EMA50 falling
                elif price < lower[i] and trend_down:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price breaks below lower Donchian (failed breakout) or EMA50 turns down
                if price < lower[i] or not trend_up:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price breaks above upper Donchian (failed breakdown) or EMA50 turns up
                if price > upper[i] or not trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50Slope_Volume"
timeframe = "4h"
leverage = 1.0