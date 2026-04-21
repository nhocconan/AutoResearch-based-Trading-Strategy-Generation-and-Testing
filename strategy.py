#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(30) breakout with 1d ATR(14) volatility filter and volume spike confirmation.
# Long when price breaks above upper Donchian in low volatility (ATR ratio < 0.8), short when breaks below lower Donchian.
# Volume > 2.0x 30-period average confirms breakout strength. ATR filter avoids false breakouts in high volatility.
# Target: 20-40 trades/year by requiring low volatility + volume + breakout alignment.
# Works in bull/bear: ATR filter adapts to market conditions, avoiding chop in ranging markets and catching trends.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Wilder's smoothing for ATR
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_14 = wilder_smooth(tr, 14)
    
    # Calculate 30-period ATR for volatility ratio (current ATR / 30-period average ATR)
    atr_30 = wilder_smooth(tr, 30)
    atr_ratio = atr_14 / atr_30  # < 1 = low volatility, > 1 = high volatility
    
    # Align ATR ratio to 12h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 30-period Donchian channels on 12h data
    high_roll = prices['high'].rolling(window=30, min_periods=30).max()
    low_roll = prices['low'].rolling(window=30, min_periods=30).min()
    upper = high_roll.values
    lower = low_roll.values
    
    # Pre-compute volume moving average (30-period)
    vol_ma = prices['volume'].rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(upper[i]) or np.isnan(lower[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volatility filter: low volatility (ATR ratio < 0.8)
        low_vol = atr_ratio_aligned[i] < 0.8
        
        # Volume confirmation: current volume > 2.0x 30-period average
        volume_confirm = volume > 2.0 * vol_ma[i]
        
        if position == 0:
            if low_vol and volume_confirm:
                # Long: price breaks above upper Donchian
                if price > upper[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below lower Donchian
                elif price < lower[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price breaks below lower Donchian (failed breakout) or volatility increases
                if price < lower[i] or atr_ratio_aligned[i] > 1.2:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price breaks above upper Donchian (failed breakdown) or volatility increases
                if price > upper[i] or atr_ratio_aligned[i] > 1.2:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian30_Breakout_1dATR_Vol_Filter"
timeframe = "12h"
leverage = 1.0