#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout + 4h EMA34 trend filter + volume spike confirmation + session filter (08-20 UTC)
# Targets 15-30 trades per year (60-120 total over 4 years) to minimize fee drag
# Camarilla R3/S3 levels act as intraday support/resistance with high breakout reliability
# 4h EMA34 ensures alignment with medium-term trend to avoid counter-trend trades
# Volume spike (2.0x 24-period average) confirms institutional participation
# Session filter reduces noise during low-liquidity hours
# Discrete position sizing 0.20 to balance exposure and minimize fee churn
# Works in both bull and bear: trend filter + volume confirmation prevent false signals

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 35:  # Need enough for EMA34
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend filter
    close_4h = pd.Series(df_4h['close'].values)
    ema_34_4h = close_4h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate Camarilla levels on 1h (based on previous bar's range)
    # Camarilla R3 = close + 1.1*(high-low)*1.1/4
    # Camarilla S3 = close - 1.1*(high-low)*1.1/4
    # Using previous bar to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range * 1.1 / 4
    s3 = prev_close - 1.1 * camarilla_range * 1.1 / 4
    
    # Calculate volume confirmation (2.0x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for volume MA and Camarilla)
    start_idx = 24
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Check for NaN values in indicators
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(volume_confirm[i]) or 
            np.isnan(r3[i]) or np.isnan(s3[i])):
            signals[i] = 0.0
            continue
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3 AND above 4h EMA34 AND volume confirm
            if (close[i] > r3[i] and 
                close[i] > ema_34_4h_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S3 AND below 4h EMA34 AND volume confirm
            elif (close[i] < s3[i] and 
                  close[i] < ema_34_4h_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Camarilla S3 OR below 4h EMA34
            if (close[i] < s3[i] or 
                close[i] < ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla R3 OR above 4h EMA34
            if (close[i] > r3[i] or 
                close[i] > ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals