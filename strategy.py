#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot (R1/S1) breakout with 1d volume spike and 1w EMA50 trend filter.
# Long when price breaks above R1 in uptrend (1w EMA50 rising), short when breaks below S1 in downtrend.
# Volume > 2x 24-period average confirms breakout strength. Uses EMA trend to filter weak trends and avoid chop.
# Target: 20-40 trades/year by requiring strong trend + volume + pivot breakout alignment.
# Works in bull/bear: EMA filter ensures only strong trends are traded, avoiding whipsaws in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels (R1, S1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    r1 = pivot + range_hl * 1.1 / 12
    s1 = pivot - range_hl * 1.1 / 12
    
    # Load 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Pre-compute volume moving average (24-period)
    vol_ma = prices['volume'].rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        # Skip if data not ready
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 2x 24-period average
        volume_confirm = volume > 2.0 * vol_ma[i]
        
        # Trend filter: EMA50 rising (bullish) or falling (bearish)
        ema_rising = ema_50_aligned[i] > ema_50_aligned[i-1] if i > 0 else False
        ema_falling = ema_50_aligned[i] < ema_50_aligned[i-1] if i > 0 else False
        
        if position == 0:
            if volume_confirm:
                # Long: price breaks above R1 with rising EMA50 (uptrend)
                if price > r1_aligned[i] and ema_rising:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below S1 with falling EMA50 (downtrend)
                elif price < s1_aligned[i] and ema_falling:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price breaks below S1 (failed breakout) or EMA50 turns down
                if price < s1_aligned[i] or not ema_rising:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price breaks above R1 (failed breakdown) or EMA50 turns up
                if price > r1_aligned[i] or not ema_falling:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0