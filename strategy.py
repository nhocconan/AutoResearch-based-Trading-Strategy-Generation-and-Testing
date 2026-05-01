#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend filter and volume confirmation.
# Uses Camarilla pivot levels from 4h timeframe to identify key intraday support/resistance.
# Breakouts above R1 or below S1 are traded in the direction of 4h EMA20 trend.
# Volume confirmation ensures breakouts have sufficient participation.
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods.
# Discrete position sizing (0.20) balances return and drawdown. Target: 60-150 trades over 4 years.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend filter
    ema_20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 4h Camarilla pivot levels
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    prev_close = df_4h['close'].shift(1).values
    
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.125 / 4
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.125 / 4
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume on 1h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    # Session filter: 08-20 UTC (precomputed for performance)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(21, 20) + 1  # 22 (for EMA20 and volume MA20)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not session_filter[i]:
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 4h EMA20 direction
        uptrend = curr_close > ema_20_4h_aligned[i]
        downtrend = curr_close < ema_20_4h_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Camarilla breakout conditions
        breakout_r1 = curr_close > camarilla_r1_aligned[i]  # Break above R1
        breakdown_s1 = curr_close < camarilla_s1_aligned[i]  # Break below S1
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R1 AND uptrend AND volume confirmation
            if breakout_r1 and uptrend and vol_confirm:
                signals[i] = 0.20
                position = 1
            # Short: Breakdown below S1 AND downtrend AND volume confirmation
            elif breakdown_s1 and downtrend and vol_confirm:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on breakdown below S1 (reversal signal)
            if curr_close < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit on breakout above R1 (reversal signal)
            if curr_close > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals