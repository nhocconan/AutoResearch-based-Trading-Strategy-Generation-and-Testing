#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA crossover with 4h trend filter and volume confirmation
# Long when 1h EMA(9) crosses above EMA(21) AND 4h EMA(50) is rising AND volume > 1.5x average
# Short when 1h EMA(9) crosses below EMA(21) AND 4h EMA(50) is falling AND volume > 1.5x average
# Uses 4h for trend direction (reduces whipsaws), 1h for entry timing
# Volume filter ensures momentum confirmation
# Target: 15-30 trades/year by requiring trend alignment + volume spike
# Works in bull/bear: 4h EMA filter avoids counter-trend trades, volume confirms strength

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA(50) for trend direction
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h EMA(50) slope (rising/falling)
    ema_slope_4h = np.diff(ema_50_4h_aligned, prepend=ema_50_4h_aligned[0])
    ema_rising_4h = ema_slope_4h > 0
    ema_falling_4h = ema_slope_4h < 0
    
    # Pre-compute 1h EMAs
    close = prices['close'].values
    ema_9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(21, n):
        # Skip if data not ready
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_9[i]) or np.isnan(ema_21[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for EMA crossover with volume confirmation and 4h trend alignment
            if ema_9[i] > ema_21[i] and ema_9[i-1] <= ema_21[i-1]:  # bullish crossover
                if ema_rising_4h[i] and volume_confirm:
                    signals[i] = 0.20
                    position = 1
            elif ema_9[i] < ema_21[i] and ema_9[i-1] >= ema_21[i-1]:  # bearish crossover
                if ema_falling_4h[i] and volume_confirm:
                    signals[i] = -0.20
                    position = -1
        
        elif position != 0:
            # Exit conditions: EMA crossover in opposite direction
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on bearish EMA crossover
                if ema_9[i] < ema_21[i] and ema_9[i-1] >= ema_21[i-1]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on bullish EMA crossover
                if ema_9[i] > ema_21[i] and ema_9[i-1] <= ema_21[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_EMA9_21_Crossover_4hEMA50_Trend_Volume"
timeframe = "1h"
leverage = 1.0