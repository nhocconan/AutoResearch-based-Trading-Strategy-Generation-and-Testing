#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
Hypothesis: For 1h timeframe, use 4h trend (EMA50) for signal direction and 1h only for entry timing.
Enter on hourly close breaking Camarilla R1/S1 levels from previous day with volume confirmation.
Use 4h EMA50 trend filter to align with higher timeframe bias. Strict conditions to limit trades to 15-30/year.
Works in bull/bear by following 4h trend; volatility-based entries reduce false breakouts.
Target: 15-30 trades/year per symbol.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

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
    
    # Calculate 4h EMA50 for trend filter (called ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema_50_4h[49] = np.mean(close_4h[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_4h)):
            ema_50_4h[i] = alpha * close_4h[i] + (1 - alpha) * ema_50_4h[i-1]
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate volume SMA(20) on 1h
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirm = volume[i] > 2.0 * vol_sma[i]
        
        # Calculate Camarilla levels from previous day's OHLC
        if i > 0:
            prev_day_idx = i - 1
            curr_date = pd.Timestamp(prices['open_time'].iloc[i]).date()
            prev_date = pd.Timestamp(prices['open_time'].iloc[prev_day_idx]).date()
            
            if curr_date != prev_date:
                # Previous bar is from previous day, use its OHLC
                ph = prices['high'].iloc[prev_day_idx]
                pl = prices['low'].iloc[prev_day_idx]
                pc = prices['close'].iloc[prev_day_idx]
                
                # Camarilla levels
                range_ = ph - pl
                r1 = pc + (range_ * 1.1 / 12)
                s1 = pc - (range_ * 1.1 / 12)
                
                if position == 0:
                    # Long: Break above R1 with uptrend (4h EMA50) and volume confirmation
                    if close[i] > r1 and close[i] > ema_50_4h_aligned[i] and volume_confirm:
                        signals[i] = 0.20
                        position = 1
                    # Short: Break below S1 with downtrend (4h EMA50) and volume confirmation
                    elif close[i] < s1 and close[i] < ema_50_4h_aligned[i] and volume_confirm:
                        signals[i] = -0.20
                        position = -1
                elif position == 1:
                    # Exit: Close crosses back below 4h EMA50
                    if close[i] < ema_50_4h_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.20
                elif position == -1:
                    # Exit: Close crosses back above 4h EMA50
                    if close[i] > ema_50_4h_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.20
            else:
                # Same day, hold current position
                if position == 1:
                    signals[i] = 0.20
                elif position == -1:
                    signals[i] = -0.20
                else:
                    signals[i] = 0.0
        else:
            # First bar, hold flat
            signals[i] = 0.0
    
    return signals