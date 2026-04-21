#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA Trend and Volume Confirmation
# Long when Williams %R < -80 (oversold) and price > 1d EMA50 and 1d volume > 1.5x 20-period average
# Short when Williams %R > -20 (overbought) and price < 1d EMA50 and 1d volume > 1.5x 20-period average
# Exit when Williams %R crosses -50 (mean reversion)
# Williams %R identifies overextended moves, EMA50 filters trend direction, volume confirms strength
# Target: 15-25 trades/year by requiring extreme readings + trend + volume

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Williams %R(14) on 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Highest High and Lowest Low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = -100 * (HH - Close) / (HH - LL)
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # Avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(williams_r[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and 1d volume
        price = prices['close'].iloc[i]
        # Get current 1d volume (same for all 12h bars within the day)
        idx_1d = i // 2  # 2 bars per day at 12h
        if idx_1d >= len(df_1d):
            idx_1d = len(df_1d) - 1
        volume_1d = df_1d['volume'].iloc[idx_1d]
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_ma = vol_ma_1d_aligned[i]
        volume_confirm = volume_1d > 1.5 * vol_ma
        
        # Trend filter: price relative to 1d EMA50
        price_above_ema = price > ema_50_aligned[i]
        price_below_ema = price < ema_50_aligned[i]
        
        if position == 0:
            if volume_confirm:
                # Long: Williams %R < -80 (oversold) and price > EMA50
                if williams_r[i] < -80 and price_above_ema:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R > -20 (overbought) and price < EMA50
                elif williams_r[i] > -20 and price_below_ema:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit when Williams %R crosses -50 (mean reversion)
            exit_signal = False
            
            if position == 1:  # long position
                if williams_r[i] > -50:
                    exit_signal = True
            
            elif position == -1:  # short position
                if williams_r[i] < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0