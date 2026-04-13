#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Williams %R extreme reversal with 1d EMA50 trend filter + volume spike confirmation
    # Long: Williams %R(14) < -80 (oversold) + price > 1d EMA50 + volume > 2.0x 20-period average
    # Short: Williams %R(14) > -20 (overbought) + price < 1d EMA50 + volume > 2.0x 20-period average
    # Exit: Williams %R crosses above -50 (long) or below -50 (short) OR price crosses 1d EMA50
    # Williams %R catches extreme reversals in both bull and bear markets
    # 1d EMA50 provides strong trend filter reducing whipsaw
    # Volume spike (2.0x) confirms institutional participation
    # Target: 20-30 trades/year for low fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 with min_periods
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])  # SMA50 as seed
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Calculate 4h Williams %R(14) with min_periods
    williams_r = np.full(n, np.nan)
    for i in range(14, n):
        highest_high = np.max(high[i-14:i+1])
        lowest_low = np.min(low[i-14:i+1])
        if highest_high != lowest_low:
            williams_r[i] = ((highest_high - close[i]) / (highest_high - lowest_low)) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # Get 4h volume for confirmation (>2.0x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    # Align 1d EMA50 to 4h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        williams_cross_up = williams_r[i] > -50 and williams_r[i-1] <= -50
        williams_cross_down = williams_r[i] < -50 and williams_r[i-1] >= -50
        
        # Trend filter from 1d EMA50
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]
        
        # Entry logic: Extreme Williams %R + trend alignment + volume confirmation
        long_entry = oversold and bullish_trend and volume_spike[i]
        short_entry = overbought and bearish_trend and volume_spike[i]
        
        # Exit logic: Williams %R crosses -50 or trend reversal
        long_exit = williams_cross_down or (close[i] < ema_1d_aligned[i])
        short_exit = williams_cross_up or (close[i] > ema_1d_aligned[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_williams_r_extreme_ema50_volume_v1"
timeframe = "4h"
leverage = 1.0