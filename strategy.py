#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation.
# In bull regime (price > 4h EMA50), go long on breakout above R3 with volume spike.
# In bear regime (price < 4h EMA50), go short on breakdown below S3 with volume spike.
# Uses Camarilla pivot levels from prior 4h for structure, 4h EMA50 for regime filter,
# and 1h volume spike for confirmation. Designed for 15-37 trades/year (60-150 over 4 years).
# Session filter (08-20 UTC) to reduce noise. Focus on BTC/ETH.

name = "1h_Camarilla_R3_S3_Breakout_4hEMA50_VolumeSpike"
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
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla pivots (prior completed 4h bar)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate prior 4h Camarilla levels (R3, S3)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    range_4h = high_4h - low_4h
    camarilla_r3 = close_4h + 1.1 * range_4h * 1.0 / 4  # R3 level
    camarilla_s3 = close_4h - 1.1 * range_4h * 1.0 / 4  # S3 level
    
    # Align Camarilla levels to 1h (wait for 4h bar to complete)
    r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Get 4h data for EMA50 trend filter
    ema_50 = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Calculate volume regime: current 1h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
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
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        ema_trend = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(r3) or np.isnan(s3) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: bull if close > 4h EMA50, bear if close < 4h EMA50
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Regime-based entry conditions
        if is_bull_regime:
            # Long: breakout above R3 with volume spike
            long_entry = (close_val > r3) and vol_spike
        else:
            long_entry = False
            
        if is_bear_regime:
            # Short: breakdown below S3 with volume spike
            short_entry = (close_val < s3) and vol_spike
        else:
            short_entry = False
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit on breakdown below S3 (failure of bullish breakout) or regime change to bear
            if close_val < s3 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit on breakout above R3 (failure of bearish breakdown) or regime change to bull
            if close_val > r3 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals