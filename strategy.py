#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 Breakout with 4h EMA50 Trend and Volume Spike
# Uses 4h for signal direction (trend and Camarilla levels from prior day)
# 1h only for entry timing precision
# Session filter (08-20 UTC) to reduce noise trades
# Discrete position sizing 0.20 to limit drawdown and fee churn
# Target: 60-150 total trades over 4 years (15-37/year)

name = "1h_Camarilla_R3_S3_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from prior 4h bar's OHLC
    prior_high = df_4h['high'].shift(1).values
    prior_low = df_4h['low'].shift(1).values
    prior_close = df_4h['close'].shift(1).values
    
    # Camarilla: PP = (H+L+C)/3, R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    pp = (prior_high + prior_low + prior_close) / 3.0
    r3 = prior_close + (prior_high - prior_low) * 1.1 / 4.0
    s3 = prior_close - (prior_high - prior_low) * 1.1 / 4.0
    
    # Align Camarilla levels to 1h (they change only when 4h bar closes)
    pp_aligned = align_htf_to_ltf(prices, df_4h, pp)
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure sufficient history for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 4h EMA trend filter
        ema_trend_up = close[i] > ema_50_4h_aligned[i]
        ema_trend_down = close[i] < ema_50_4h_aligned[i]
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Price > R3, 4h EMA50 uptrend, volume confirm, in session
            if price > r3_aligned[i] and ema_trend_up and vol_confirm:
                signals[i] = 0.20
                position = 1
            # Short entry: Price < S3, 4h EMA50 downtrend, volume confirm, in session
            elif price < s3_aligned[i] and ema_trend_down and vol_confirm:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on retracement to PP or below S3
            if price < pp_aligned[i] or price < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short - exit on retracement to PP or above R3
            if price > pp_aligned[i] or price > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals