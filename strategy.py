#!/usr/bin/env python3
"""
Hypothesis: 4h-based strategy combining Camarilla pivot levels (S1/R1) with 1d EMA(34) trend filter,
volume confirmation, and ATR(14) stoploss. Uses Camarilla levels as dynamic support/resistance
for mean-reversion entries in ranging markets and breakout confirmations in trending markets.
Targets 20-50 trades/year to minimize fee drag while maintaining robustness across market regimes.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA(34) trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * 2/35) + (ema_34_1d[i-1] * 33/35)
    
    # Align 1d EMA to 4h timeframe
    ema_34_1d_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h ATR(14)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[:14])
        else:
            atr[i] = (tr[i] * 1/14) + (atr[i-1] * 13/14)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Calculate Camarilla levels from 1d OHLC
    # Camarilla: based on previous day's range
    # S1 = C - (H-L)*1.08/2, R1 = C + (H-L)*1.08/2
    # Using previous day's data to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = np.roll(close_1d, 1)  # previous day close
    high_1d_prev = np.roll(high_1d, 1)    # previous day high
    low_1d_prev = np.roll(low_1d, 1)      # previous day low
    
    # Calculate Camarilla S1 and R1 for each day
    camarilla_s1_1d = np.full(len(close_1d), np.nan)
    camarilla_r1_1d = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):  # start from 1 to avoid look-ahead on day 0
        if not (np.isnan(high_1d_prev[i]) or np.isnan(low_1d_prev[i]) or np.isnan(close_1d_prev[i])):
            range_1d = high_1d_prev[i] - low_1d_prev[i]
            camarilla_s1_1d[i] = close_1d_prev[i] - range_1d * 1.08 / 2
            camarilla_r1_1d[i] = close_1d_prev[i] + range_1d * 1.08 / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_s1_4h = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    camarilla_r1_4h = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 1)  # need EMA, ATR, volume MA, and Camarilla
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_1d_4h[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_s1_4h[i]) or np.isnan(camarilla_r1_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below 1d EMA34
        trend_up = close[i] > ema_34_1d_4h[i]
        trend_down = close[i] < ema_34_1d_4h[i]
        
        if position == 0:
            # Long entry: price near S1 support with bullish bias
            # Enter when price touches or goes slightly below S1, then reverses up
            if (close[i] <= camarilla_s1_4h[i] + 0.1 * atr[i] and  # near S1
                close[i] > camarilla_s1_4h[i] - 0.2 * atr[i] and   # not too far below
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: price near R1 resistance with bearish bias
            # Enter when price touches or goes slightly above R1, then reverses down
            elif (close[i] >= camarilla_r1_4h[i] - 0.1 * atr[i] and  # near R1
                  close[i] < camarilla_r1_4h[i] + 0.2 * atr[i] and   # not too far above
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price reaches R1 resistance or ATR-based stop
            if close[i] >= camarilla_r1_4h[i] - 0.1 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches S1 support or ATR-based stop
            if close[i] <= camarilla_s1_4h[i] + 0.1 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_S1R1_1dEMA34_VolumeFilter"
timeframe = "4h"
leverage = 1.0