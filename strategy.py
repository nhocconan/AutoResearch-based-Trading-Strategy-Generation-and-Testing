#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Uses 12h Camarilla pivot levels (R3/S3) for breakout entries, filtered by 1d EMA34 trend direction.
# Volume confirmation requires current volume > 1.5x 20-period average to avoid false breakouts.
# Works in bull markets (buy R3 breakout with uptrend) and bear markets (sell S3 breakdown with downtrend).
# Discrete position sizing (0.25) balances return and drawdown. Target: 50-150 trades over 4 years.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(34, 20) + 1  # 35
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Calculate 12h Camarilla levels (R3, S3) using previous bar's OHLC
        if i == 0:
            signals[i] = 0.0
            continue
            
        prev_close = close[i-1]
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_range = prev_high - prev_low
        
        # Camarilla levels
        R3 = prev_close + (prev_range * 1.1 / 4)
        S3 = prev_close - (prev_range * 1.1 / 4)
        
        # Trend filter: 1d EMA34 direction
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = curr_volume > (vol_avg_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above R3 AND uptrend AND volume confirmation
            if curr_close > R3 and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 AND downtrend AND volume confirmation
            elif curr_close < S3 and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on break below S3 (reversal signal)
            if curr_close < S3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on break above R3 (reversal signal)
            if curr_close > R3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals