#!/usr/bin/env python3
"""
6h_Camarilla_R4S4_Breakout_1dTrend_VolumeConfirm
Hypothesis: Camarilla R4/S4 breakout on 6h with 1d EMA34 trend filter and volume confirmation (>1.8x average volume). 
In bull markets: price breaks above R4 with 1d uptrend and high volume → long.
In bear markets: price breaks below S4 with 1d downtrend and high volume → short.
Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 trades over 4 years (12-37/year) on 6h timeframe.
Requires BTC/ETH edge via 1d trend and volume filters; avoids SOL-only bias by requiring trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for EMA and volume
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20 for volume SMA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Calculate Camarilla levels using previous day's OHLC (1d lookback)
        if i < 24:  # Need at least 24 6h bars for 1 day (4*6h=24h)
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        # Get previous day's OHLC (24 lookback for 6h timeframe)
        lookback_start = i - 24
        if lookback_start < 0:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        # Previous day's high, low, close
        prev_high = np.max(high[lookback_start:i])
        prev_low = np.min(low[lookback_start:i])
        prev_close = close[i-1]  # Previous bar's close
        
        # Camarilla levels calculation
        range_val = prev_high - prev_low
        if range_val <= 0:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        # Camarilla R4 and S4 levels
        r4 = prev_close + range_val * 1.1 / 2
        s4 = prev_close - range_val * 1.1 / 2
        
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_34_1d_aligned[i]
        
        # Skip if any data not ready
        if np.isnan(r4) or np.isnan(s4) or np.isnan(ema_val) or np.isnan(avg_vol):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 1.8x average volume
        volume_confirmed = vol > 1.8 * avg_vol
        
        # Long logic: price breaks above R4 with 1d uptrend and volume confirmation
        long_condition = (close_val > r4) and (close_val > ema_val) and volume_confirmed
        # Short logic: price breaks below S4 with 1d downtrend and volume confirmation
        short_condition = (close_val < s4) and (close_val < ema_val) and volume_confirmed
        
        # Exit logic: trend reversal or opposite breakout
        exit_long = (close_val < ema_val) or (close_val < s4)
        exit_short = (close_val > ema_val) or (close_val > r4)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Camarilla_R4S4_Breakout_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0