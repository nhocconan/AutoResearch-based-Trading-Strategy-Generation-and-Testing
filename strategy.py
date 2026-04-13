#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h strategy using 1d Camarilla pivot levels (H3/L3) breakout with 1w trend filter (EMA34)
    # Long: price breaks above H3 AND 1w EMA34 rising (bullish trend) AND volume > 1.5x avg
    # Short: price breaks below L3 AND 1w EMA34 falling (bearish trend) AND volume > 1.5x avg
    # Exit: price returns to Pivot point or opposite breakout
    # Using 12h timeframe for low trade frequency, Camarilla for structure, 1w EMA for regime, volume for confirmation.
    # Discrete position sizing (0.30) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (H3, L3, Pivot)
    # Based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    pivot = typical_price
    h3 = close_1d + (range_1d * 1.1 / 4.0)  # H3 = Close + 1.1*(Range)/4
    l3 = close_1d - (range_1d * 1.1 / 4.0)  # L3 = Close - 1.1*(Range)/4
    
    # Align daily Camarilla levels to 12h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Get weekly data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Calculate weekly EMA34
    close_1w = df_1w['close'].values
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA34 to 12h
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # EMA34 direction (1 = rising, -1 = falling)
    ema_dir = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(ema_34_aligned[i]) and not np.isnan(ema_34_aligned[i-1]):
            if ema_34_aligned[i] > ema_34_aligned[i-1]:
                ema_dir[i] = 1
            elif ema_34_aligned[i] < ema_34_aligned[i-1]:
                ema_dir[i] = -1
            else:
                ema_dir[i] = ema_dir[i-1]
        else:
            ema_dir[i] = 0
    
    # Get 12h volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(ema_dir[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: EMA34 direction
        bullish_trend = ema_dir[i] == 1
        bearish_trend = ema_dir[i] == -1
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: Camarilla breakout + trend + volume
        long_entry = (close[i] > h3_aligned[i]) and bullish_trend and vol_confirm
        short_entry = (close[i] < l3_aligned[i]) and bearish_trend and vol_confirm
        
        # Exit logic: price returns to pivot or opposite breakout
        long_exit = (close[i] < pivot_aligned[i]) or (close[i] < l3_aligned[i])
        short_exit = (close[i] > pivot_aligned[i]) or (close[i] > h3_aligned[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.30
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.30
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_1w_camarilla_ema_volume_v1"
timeframe = "12h"
leverage = 1.0