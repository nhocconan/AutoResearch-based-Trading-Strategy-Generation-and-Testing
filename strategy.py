#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w trend filter and volume confirmation.
# Uses 1w EMA50 as trend filter and 1w ATR for volatility-based stoploss.
# Works in bull (buy R3 breakout with uptrend) and bear (sell S3 breakdown with downtrend).
# Discrete position sizing 0.25 balances return and drawdown. Target: 50-150 trades over 4 years.

name = "12h_Camarilla_R3_S3_Breakout_1wEMA50_Trend_VolumeConfirm_v1"
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
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1w Camarilla pivot levels (R3, S3) - strong breakout levels
    # Camarilla: based on previous week's high, low, close
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    prev_weekly_close = df_1w['close'].shift(1).values
    
    camarilla_r3_1w = prev_weekly_close + (prev_weekly_high - prev_weekly_low) * 1.1 / 4
    camarilla_s3_1w = prev_weekly_close - (prev_weekly_high - prev_weekly_low) * 1.1 / 4
    
    camarilla_r3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3_1w)
    camarilla_s3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3_1w)
    
    # 12h Camarilla pivot levels (R3, S3) for breakout
    # Based on previous 12h bar's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_r3_12h = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3_12h = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume on 12h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 20) + 1  # 51 (for EMA50 and volume MA)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r3_1w_aligned[i]) or
            np.isnan(camarilla_s3_1w_aligned[i]) or
            np.isnan(camarilla_r3_12h[i]) or
            np.isnan(camarilla_s3_12h[i]) or
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
        
        # 12h Camarilla breakout conditions
        breakout_r3 = curr_high > camarilla_r3_12h[i]  # Break above 12h R3
        breakdown_s3 = curr_low < camarilla_s3_12h[i]  # Break below 12h S3
        
        # Weekly Camarilla R3/S3 confirmation
        confirm_r3 = curr_close > camarilla_r3_1w_aligned[i]  # Confirm above weekly R3
        confirm_s3 = curr_close < camarilla_s3_1w_aligned[i]  # Confirm below weekly S3
        
        if position == 0:  # Flat - look for new entries
            # Long: 12h R3 breakout AND weekly R3 confirmation AND uptrend AND volume confirmation
            if breakout_r3 and confirm_r3 and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: 12h S3 breakdown AND weekly S3 confirmation AND downtrend AND volume confirmation
            elif breakdown_s3 and confirm_s3 and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on 12h S3 breakdown (reversal signal)
            if curr_low < camarilla_s3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on 12h R3 breakout (reversal signal)
            if curr_high > camarilla_r3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals