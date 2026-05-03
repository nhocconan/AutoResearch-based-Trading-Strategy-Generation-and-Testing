#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike.
# Long when price breaks above Camarilla R3 in bull trend (close > 4h EMA50) with volume > 2.0x 20-period MA.
# Short when price breaks below Camarilla S3 in bear trend (close < 4h EMA50) with volume spike.
# Session filter: only trade between 08:00-20:00 UTC to avoid low-liquidity hours.
# Uses discrete position sizing (0.20) to minimize fee churn and control drawdown.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.

name = "1h_Camarilla_R3S3_4hEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08:00-20:00 UTC) - avoid .hour on datetime64
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA trend filter and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align EMA to 1h timeframe (wait for completed 4h bar)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels for each 4h bar, then align
    if len(df_4h) >= 1:
        H_4h = df_4h['high'].values
        L_4h = df_4h['low'].values
        C_4h = df_4h['close'].values
        
        # Typical price = (H+L+C)/3
        P_4h = (H_4h + L_4h + C_4h) / 3.0
        range_4h = H_4h - L_4h
        
        # Camarilla R3 and S3
        camarilla_R3_4h = P_4h + (range_4h * 1.1 / 4.0)
        camarilla_S3_4h = P_4h - (range_4h * 1.1 / 4.0)
        
        # Align to 1h timeframe (wait for completed 4h bar)
        camarilla_R3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_R3_4h)
        camarilla_S3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_S3_4h)
    else:
        camarilla_R3_aligned = np.full(n, np.nan)
        camarilla_S3_aligned = np.full(n, np.nan)
    
    # Volume regime: current 1h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_4h_aligned[i]
        r3_level = camarilla_R3_aligned[i]
        s3_level = camarilla_S3_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Camarilla breakout conditions (using current bar's levels)
        breakout_up = close_val > r3_level
        breakout_down = close_val < s3_level
        
        # Entry logic
        if position == 0:
            if is_bull_trend and breakout_up and vol_spike:
                signals[i] = 0.20
                position = 1
            elif is_bear_trend and breakout_down and vol_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Camarilla S3 break OR trend reversal
            if close_val < s3_level or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Camarilla R3 break OR trend reversal
            if close_val > r3_level or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals