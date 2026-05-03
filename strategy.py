#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation.
# Long when price breaks above Camarilla R1 level in bull trend (close > 4h EMA50) with volume > 2.0x 20-period MA.
# Short when price breaks below Camarilla S1 level in bear trend (close < 4h EMA50) with volume spike.
# Uses 4h/1d for signal direction, 1h only for entry timing to reduce trade frequency.
# Session filter (08-20 UTC) to avoid low-liquidity periods.
# Discrete position sizing (0.20) to minimize fee churn. Target: 60-150 total trades over 4 years.

name = "1h_Camarilla_R1S1_4hEMA50_Volume_Session"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Camarilla levels (based on previous 1d bar's range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar
    prev_1d_high = df_1d['high'].values
    prev_1d_low = df_1d['low'].values
    prev_1d_close = df_1d['close'].values
    
    camarilla_r1 = prev_1d_close + (prev_1d_high - prev_1d_low) * 1.1 / 12
    camarilla_s1 = prev_1d_close - (prev_1d_high - prev_1d_low) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (no additional delay needed as these are based on completed 1d bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume regime: current 1h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_4h_aligned[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Camarilla breakout conditions
        breakout_r1 = close_val > r1_level
        breakout_s1 = close_val < s1_level
        
        # Entry logic
        if position == 0:
            if is_bull_trend and breakout_r1 and vol_spike:
                signals[i] = 0.20
                position = 1
            elif is_bear_trend and breakout_s1 and vol_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 level OR trend reversal
            if close_val < s1_level or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above R1 level OR trend reversal
            if close_val > r1_level or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals