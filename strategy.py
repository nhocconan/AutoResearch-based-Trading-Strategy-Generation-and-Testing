#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot (R1/S1) breakout with 1w EMA(34) trend filter and volume spike confirmation.
# Long when price breaks above R1 in uptrend (1w EMA34 > prior EMA34), short when breaks below S1 in downtrend.
# Volume > 1.5x 20-period average confirms breakout strength. Uses weekly EMA trend to avoid chop.
# Target: 20-40 trades/year by requiring strong trend + volume + pivot breakout alignment.
# Works in bull/bear: EMA trend filter ensures only clear trends are traded, avoiding whipsaws in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 12h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Load 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R1, S1) from prior 1d bar
    # Typical Price = (H + L + C) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3
    # Camarilla R1 = Close + 1.1 * (High - Low) / 12
    # Camarilla S1 = Close - 1.1 * (High - Low) / 12
    r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align pivot levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA warmup
        # Skip if data not ready
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Trend filter: upward trending EMA (current > previous)
        ema_rising = ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]
        ema_falling = ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1]
        
        if position == 0:
            if volume_confirm:
                # Long: price breaks above R1 in uptrend
                if price > r1_aligned[i] and ema_rising:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below S1 in downtrend
                elif price < s1_aligned[i] and ema_falling:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price breaks below S1 (failed breakout) or trend turns down
                if price < s1_aligned[i] or ema_falling:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price breaks above R1 (failed breakdown) or trend turns up
                if price > r1_aligned[i] or ema_rising:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1wEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0