#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot (R1/S1) breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above R1 in uptrend (1d EMA34 rising), short when breaks below S1 in downtrend.
# Volume > 2x 20-period average confirms breakout strength. EMA34 slope filters weak trends.
# Target: 25-50 trades/year by requiring strong trend + volume + pivot breakout alignment.
# Works in bull/bear: EMA34 filter ensures only trending markets are traded, avoiding whipsaws in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_slope = ema_34 - np.roll(ema_34, 1)  # slope: current - previous
    ema_34_slope[0] = 0  # first period
    
    # Align EMA34 slope to 4h timeframe
    ema_34_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_34_slope)
    
    # Calculate Camarilla pivot levels from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    R1 = close_1d + range_1d * 1.1 / 12
    S1 = close_1d - range_1d * 1.1 / 12
    
    # Align levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if data not ready
        if np.isnan(ema_34_slope_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 2x 20-period average
        volume_confirm = volume > 2.0 * vol_ma[i]
        
        # Trend filter: EMA34 slope positive for long, negative for short
        uptrend = ema_34_slope_aligned[i] > 0
        downtrend = ema_34_slope_aligned[i] < 0
        
        if position == 0:
            if volume_confirm:
                # Long: price breaks above R1 in uptrend
                if price > R1_aligned[i] and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below S1 in downtrend
                elif price < S1_aligned[i] and downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price breaks below S1 (failed breakout) or trend turns down
                if price < S1_aligned[i] or ema_34_slope_aligned[i] < 0:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price breaks above R1 (failed breakdown) or trend turns up
                if price > R1_aligned[i] or ema_34_slope_aligned[i] > 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0