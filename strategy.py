#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot (R1/S1) breakout with 1d EMA34 trend filter and volume confirmation.
# In uptrend (price > 1d EMA34): long on break above R1, short on break below S1.
# In downtrend (price < 1d EMA34): short on break below S1, long on break above R1.
# Uses volume > 1.5x 20-period average for confirmation.
# Target: 25-50 trades/year by requiring trend alignment + pivot breakout + volume spike.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if data not ready
        if np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels from previous day
        # Need high, low, close from previous completed day
        prev_day_idx = i // 96  # 96 = 24*4 (4h bars per day)
        if prev_day_idx < 1:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get previous day's OHLC from 1d data
        if prev_day_idx >= len(df_1d):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ph = df_1d['high'].iloc[prev_day_idx - 1]
        pl = df_1d['low'].iloc[prev_day_idx - 1]
        pc = df_1d['close'].iloc[prev_day_idx - 1]
        
        # Camarilla levels
        range_ = ph - pl
        r1 = pc + (range_ * 1.1 / 12)
        s1 = pc - (range_ * 1.1 / 12)
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Trend filter: price vs 1d EMA34
        is_uptrend = price > ema_34_aligned[i]
        is_downtrend = price < ema_34_aligned[i]
        
        if position == 0:
            if volume_confirm:
                if is_uptrend and price > r1:
                    signals[i] = 0.25
                    position = 1
                elif is_downtrend and price < s1:
                    signals[i] = -0.25
                    position = -1
                # Counter-trend entries in strong momentum
                elif not is_uptrend and price > r1:
                    signals[i] = 0.25
                    position = 1
                elif not is_downtrend and price < s1:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on reversal below S1 or if trend changes against position
                if price < s1 or (is_downtrend and price < ema_34_aligned[i]):
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on reversal above R1 or if trend changes against position
                if price > r1 or (is_uptrend and price > ema_34_aligned[i]):
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