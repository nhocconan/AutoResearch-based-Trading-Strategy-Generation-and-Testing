#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: 12h Camarilla R3/S3 breakout in direction of 1d EMA34 trend, confirmed by volume spike (>2x 20-bar MA). Exits via opposite Camarilla level (S3/R3) or ATR trailing stop. Designed for low frequency (12-37 trades/year) to avoid fee drag. Uses discrete sizing (0.25) and works in bull/bear via 1d trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR(14) for stoploss calculation
    atr_period = 14
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    if n >= atr_period:
        atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, n):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size
    
    # Track Camarilla levels from previous 1d bar
    camarilla_R3 = 0.0
    camarilla_S3 = 0.0
    camarilla_R4 = 0.0
    camarilla_S4 = 0.0
    
    # Warmup: max of calculations
    start_idx = max(20, 34, 20, atr_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Determine 1d trend: bullish if close > EMA34, bearish if close < EMA34
        bullish_1d = close_val > ema_34_val
        bearish_1d = close_val < ema_34_val
        
        # Update Camarilla levels from completed 1d bar (only when 1d bar closes)
        # Check if current 15m bar is the first bar of a new 1d bar
        # Since we're on 12h timeframe, we check if we've moved to a new day
        if i >= 2:  # Need at least 2 bars to check for day change
            prev_day = pd.Timestamp(prices['open_time'].iloc[i-1]).date()
            curr_day = pd.Timestamp(prices['open_time'].iloc[i]).date()
            if prev_day != curr_day:
                # New day started, calculate Camarilla from previous day's OHLC
                prev_idx = i - 1
                while prev_idx >= 0 and pd.Timestamp(prices['open_time'].iloc[prev_idx]).date() == prev_day:
                    prev_idx -= 1
                prev_idx += 1  # First bar of previous day
                
                if prev_idx < i and prev_idx >= 0:
                    day_high = np.max(high[prev_idx:i])
                    day_low = np.min(low[prev_idx:i])
                    day_close = close[i-1]  # Close of previous day
                    
                    # Camarilla levels
                    range_val = day_high - day_low
                    camarilla_R4 = day_close + range_val * 1.1 / 2
                    camarilla_R3 = day_close + range_val * 1.1 / 4
                    camarilla_S3 = day_close - range_val * 1.1 / 4
                    camarilla_S4 = day_close - range_val * 1.1 / 2
        
        # Entry conditions
        long_entry = (close_val > camarilla_R3) and bullish_1d and vol_spike
        short_entry = (close_val < camarilla_S3) and bearish_1d and vol_spike
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit if price breaks below S3 (opposite level) or ATR stop
            if close_val < camarilla_S3:
                exit_long = True
            elif high_val > camarilla_R3 and close_val < (camarilla_R3 - 0.5 * atr_val):
                exit_long = True
        elif position == -1:
            # Exit if price breaks above R3 (opposite level) or ATR stop
            if close_val > camarilla_R3:
                exit_short = True
            elif low_val < camarilla_S3 and close_val > (camarilla_S3 + 0.5 * atr_val):
                exit_short = True
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0