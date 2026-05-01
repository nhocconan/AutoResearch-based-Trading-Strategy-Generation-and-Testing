#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R4/S4 breakout with 1w trend filter and volume confirmation.
# Uses 1w EMA50 as trend filter to capture major trend direction and avoid counter-trend trades.
# Camarilla R4/S4 are strong breakout levels (beyond R3/S3) that indicate sustained momentum.
# Works in bull (buy R4 breakout with uptrend) and bear (sell S4 breakdown with downtrend).
# Discrete position sizing 0.25 balances return and drawdown. Target: 50-150 trades over 4 years.

name = "6h_Camarilla_R4_S4_Breakout_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1w Camarilla pivot levels (R4, S4) - strong breakout levels
    # Camarilla: based on previous week's high, low, close
    # R4 = close + (high - low) * 1.1/2
    # S4 = close - (high - low) * 1.1/2
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    prev_weekly_close = df_1w['close'].shift(1).values
    
    camarilla_r4_1w = prev_weekly_close + (prev_weekly_high - prev_weekly_low) * 1.1 / 2
    camarilla_s4_1w = prev_weekly_close - (prev_weekly_high - prev_weekly_low) * 1.1 / 2
    
    camarilla_r4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4_1w)
    camarilla_s4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4_1w)
    
    # 6h Camarilla pivot levels (R4, S4) for breakout
    # Based on previous 6h bar's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_r4_6h = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s4_6h = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume on 6h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 20) + 1  # 51 (for EMA50 and volume MA)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r4_1w_aligned[i]) or
            np.isnan(camarilla_s4_1w_aligned[i]) or
            np.isnan(camarilla_r4_6h[i]) or
            np.isnan(camarilla_s4_6h[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: 1w EMA50 direction
        uptrend = curr_close > ema_50_1w_aligned[i]
        downtrend = curr_close < ema_50_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 6h Camarilla breakout conditions
        breakout_r4 = curr_high > camarilla_r4_6h[i]  # Break above 6h R4
        breakdown_s4 = curr_low < camarilla_s4_6h[i]  # Break below 6h S4
        
        # Weekly Camarilla R4/S4 confirmation
        confirm_r4 = curr_close > camarilla_r4_1w_aligned[i]  # Confirm above weekly R4
        confirm_s4 = curr_close < camarilla_s4_1w_aligned[i]  # Confirm below weekly S4
        
        if position == 0:  # Flat - look for new entries
            # Long: 6h R4 breakout AND weekly R4 confirmation AND uptrend AND volume confirmation
            if breakout_r4 and confirm_r4 and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: 6h S4 breakdown AND weekly S4 confirmation AND downtrend AND volume confirmation
            elif breakdown_s4 and confirm_s4 and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on 6h S4 breakdown (reversal signal)
            if curr_low < camarilla_s4_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on 6h R4 breakout (reversal signal)
            if curr_high > camarilla_r4_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals