#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike.
# Long when price breaks above R3 with 1d EMA34 uptrend and volume > 2x 20-period MA.
# Short when price breaks below S3 with 1d EMA34 downtrend and volume > 2x 20-period MA.
# Camarilla levels provide precise intraday support/resistance; EMA34 filters trend;
# volume spike confirms institutional participation. Works in bull via longs and bear via shorts.
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.30.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA-34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume regime: current 4h volume > 2x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Camarilla levels from previous day (using prior day's high, low, close)
        if i < 96:  # Need at least 4 prior 4h bars (1 day) to calculate
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Get prior day's OHLC (24h ago = 6 * 4h bars)
        idx_prev_day = i - 6
        if idx_prev_day < 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        high_prev = high[idx_prev_day]
        low_prev = low[idx_prev_day]
        close_prev = close[idx_prev_day]
        
        # Calculate Camarilla levels
        range_prev = high_prev - low_prev
        if range_prev <= 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        R3 = close_prev + range_prev * 1.1 / 4
        S3 = close_prev - range_prev * 1.1 / 4
        
        close_val = close[i]
        ema_trend = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend: price above/below EMA34
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            # Long: break above R3 AND uptrend AND volume spike
            if close_val > R3 and is_uptrend and vol_spike:
                signals[i] = 0.30
                position = 1
            # Short: break below S3 AND downtrend AND volume spike
            elif close_val < S3 and is_downtrend and vol_spike:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price below S3 OR trend reverses OR volume drops
            if close_val < S3 or not is_uptrend or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price above R3 OR trend reverses OR volume drops
            if close_val > R3 or not is_downtrend or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals