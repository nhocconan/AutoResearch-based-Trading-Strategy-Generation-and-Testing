#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike
Hypothesis: On 1d timeframe, Camarilla R1/S1 breakouts with 1w EMA50 trend filter and volume confirmation (>2.0x 20-bar avg) capture institutional breakouts in both bull and bear markets. Uses weekly timeframe for trend to reduce noise and avoid lower timeframe whipsaws. Targets 15-25 trades/year to minimize fee drag while maintaining edge via trend filter and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Get 1w data for HTF trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume average (20-period) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(30, 50, 20)  # 1d lookback, 1w EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_50_val = ema_50_1w_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = vol_val > 2.0 * vol_ma_val
        
        # Calculate Camarilla levels for current 1d bar (using previous day's OHLC)
        if i >= 1:  # need previous day
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            prev_close = close_1d[i-1]
            
            # Camarilla pivot levels
            pivot = (prev_high + prev_low + prev_close) / 3.0
            range_val = prev_high - prev_low
            
            # Resistance and Support levels
            r1 = pivot + (range_val * 1.1 / 12)
            s1 = pivot - (range_val * 1.1 / 12)
            
            if position == 0:
                # Long: price breaks above R1 with uptrend (close > EMA50) and volume confirmation
                long_signal = (high_val > r1) and (close_val > ema_50_val) and volume_confirmed
                # Short: price breaks below S1 with downtrend (close < EMA50) and volume confirmation
                short_signal = (low_val < s1) and (close_val < ema_50_val) and volume_confirmed
                
                if long_signal:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close_val
                elif short_signal:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close_val
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Long: hold position
                signals[i] = 0.25
                # Exit conditions:
                # 1. Trend reversal: close crosses below EMA50
                if close_val < ema_50_val:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                # 2. Price re-enters Camarilla range (mean reversion exit)
                elif low_val <= pivot and high_val >= pivot:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
            elif position == -1:
                # Short: hold position
                signals[i] = -0.25
                # Exit conditions:
                # 1. Trend reversal: close crosses above EMA50
                if close_val > ema_50_val:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                # 2. Price re-enters Camarilla range (mean reversion exit)
                elif low_val <= pivot and high_val >= pivot:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0