#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation.
# Uses 4h HTF for EMA50 trend and 1d HTF for Camarilla levels (previous day) to avoid look-ahead.
# Long when price breaks above Camarilla R3 AND price > 4h EMA50 AND volume > 2.0x 20-bar average.
# Short when price breaks below Camarilla S3 AND price < 4h EMA50 AND volume > 2.0x 20-bar average.
# Exit when price crosses Camarilla H3/L3 midline.
# Discrete position sizing (0.20) to limit drawdown and fee churn.
# Session filter: 08-20 UTC to reduce noise trades.
# Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe.
# Works in bull/bear via 4h EMA50 trend filter and volume confirmation to avoid false breakouts.

name = "1h_Camarilla_R3S3_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE before loop for Camarilla levels (previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Use previous day's OHLC for Camarilla calculation (shifted by 1 to avoid look-ahead)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: based on previous day's range
    rang = prev_high - prev_low
    camarilla_r3 = prev_close + rang * 1.1 / 4
    camarilla_s3 = prev_close - rang * 1.1 / 4
    camarilla_h3 = prev_close + rang * 1.1 / 6
    camarilla_l3 = prev_close - rang * 1.1 / 6
    camarilla_h3_l3_mid = (camarilla_h3 + camarilla_l3) / 2
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h3_l3_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_l3_mid)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_h3_l3_mid_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not in_session[i]:
            signals[i] = 0.0
            if position == 1:
                position = 0
            elif position == -1:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above Camarilla R3, uptrend (price > 4h EMA50), volume confirmation
            if (curr_high > camarilla_r3_aligned[i] and 
                curr_close > ema_50_4h_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: break below Camarilla S3, downtrend (price < 4h EMA50), volume confirmation
            elif (curr_low < camarilla_s3_aligned[i] and 
                  curr_close < ema_50_4h_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: Camarilla H3/L3 midline cross
            if curr_close < camarilla_h3_l3_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit condition: Camarilla H3/L3 midline cross
            if curr_close > camarilla_h3_l3_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals