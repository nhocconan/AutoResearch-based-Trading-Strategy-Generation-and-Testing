#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter (EMA50) and volume spike confirmation.
# In bull regime (price > 4h EMA50), go long on break above Camarilla R3 with volume spike.
# In bear regime (price < 4h EMA50), go short on break below Camarilla S3 with volume spike.
# Uses 4h/1d for signal direction, 1h only for entry timing. Target: 15-37 trades/year.

name = "1h_Camarilla_R3S3_Breakout_4hTrend_VolumeSpike"
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
    
    # Get 4h data for trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 trend filter
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Calculate Camarilla levels from previous 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels: based on previous day's range
    R3 = close_4h + (high_4h - low_4h) * 1.1 / 4
    S3 = close_4h - (high_4h - low_4h) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (wait for 4h bar to close)
    R3_aligned = align_htf_to_ltf(prices, df_4h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_4h, S3)
    
    # Calculate volume regime: current 1h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if outside session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Get current values
        close_val = close[i]
        ema_trend = ema_50_aligned[i]
        r3_level = R3_aligned[i]
        s3_level = S3_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(close_val) or np.isnan(ema_trend) or np.isnan(r3_level) or np.isnan(s3_level):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: bull if close > 4h EMA50, bear if close < 4h EMA50
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Generate signals
        if position == 0:
            if is_bull_regime and close_val > r3_level and vol_spike:
                signals[i] = 0.20
                position = 1
            elif is_bear_regime and close_val < s3_level and vol_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit on close below 4h EMA50 (trend change) or loss of momentum
            if close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit on close above 4h EMA50 (trend change) or loss of momentum
            if close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals