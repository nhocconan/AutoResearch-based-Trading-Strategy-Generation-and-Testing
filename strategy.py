#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h trend filter and volume confirmation.
# Uses 4h EMA34 as trend filter and 4h ATR for volatility filter.
# Works in bull (buy R3 breakout with uptrend) and bear (sell S3 breakdown with downtrend).
# Session filter: 08-20 UTC to reduce noise trades. Discrete position sizing 0.20.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.

name = "1h_Camarilla_R3_S3_Breakout_4hEMA34_Trend_VolumeConfirm_v1"
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
    
    # Calculate 4h EMA34 for trend filter
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 4h ATR14 for volatility filter
    tr1 = df_4h['high'] - df_4h['low']
    tr2 = np.abs(df_4h['high'] - df_4h['close'].shift(1))
    tr3 = np.abs(df_4h['low'] - df_4h['close'].shift(1))
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_4h = tr_4h.rolling(window=14, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Calculate 4h Camarilla pivot levels (R3, S3)
    prev_daily_high = df_4h['high'].shift(1).values
    prev_daily_low = df_4h['low'].shift(1).values
    prev_daily_close = df_4h['close'].shift(1).values
    
    camarilla_r3_4h = prev_daily_close + (prev_daily_high - prev_daily_low) * 1.1 / 4
    camarilla_s3_4h = prev_daily_close - (prev_daily_high - prev_daily_low) * 1.1 / 4
    
    camarilla_r3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    
    # 1h Camarilla pivot levels (R3, S3) for breakout
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_r3_1h = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3_1h = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume on 1h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(34, 20, 14) + 1  # 35 (for EMA34, volume MA, ATR)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(atr_14_4h_aligned[i]) or
            np.isnan(camarilla_r3_4h_aligned[i]) or
            np.isnan(camarilla_s3_4h_aligned[i]) or
            np.isnan(camarilla_r3_1h[i]) or
            np.isnan(camarilla_s3_1h[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        if not session_filter[i]:
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: 4h EMA34 direction
        uptrend = curr_close > ema_34_4h_aligned[i]
        downtrend = curr_close < ema_34_4h_aligned[i]
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr_14_4h_aligned[i] > 0
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1h Camarilla breakout conditions
        breakout_r3 = curr_high > camarilla_r3_1h[i]  # Break above 1h R3
        breakdown_s3 = curr_low < camarilla_s3_1h[i]  # Break below 1h S3
        
        # 4h Camarilla R3/S3 confirmation
        confirm_r3 = curr_close > camarilla_r3_4h_aligned[i]  # Confirm above 4h R3
        confirm_s3 = curr_close < camarilla_s3_4h_aligned[i]  # Confirm below 4h S3
        
        if position == 0:  # Flat - look for new entries
            # Long: 1h R3 breakout AND 4h R3 confirmation AND uptrend AND volume confirmation AND vol filter
            if breakout_r3 and confirm_r3 and uptrend and vol_confirm and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: 1h S3 breakdown AND 4h S3 confirmation AND downtrend AND volume confirmation AND vol filter
            elif breakdown_s3 and confirm_s3 and downtrend and vol_confirm and vol_filter:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on 1h S3 breakdown (reversal signal)
            if curr_low < camarilla_s3_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit on 1h R3 breakout (reversal signal)
            if curr_high > camarilla_r3_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals