#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 12h EMA50 trend filter and volume spike confirmation.
# Williams %R identifies overbought/oversold conditions; extreme readings (<-90 or >-10) signal potential reversals.
# 12h EMA50 provides trend alignment to trade with the higher timeframe momentum.
# Volume spike (>2x 20-period average) confirms strong participation at reversal points.
# Session filter (08-20 UTC) focuses on active trading hours to reduce noise.
# Discrete position sizing (0.25) balances risk and return. Target: 12-25 trades/year.

name = "6h_WilliamsR_Extreme_12hEMA50_VolumeSpike_Session_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h data
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams %R(14) on 6h data
    if n >= 14:
        hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
        ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
        # Avoid division by zero
        denom = hh - ll
        denom = np.where(denom == 0, 1e-10, denom)
        williams_r = -100 * ((hh - close) / denom)
    else:
        williams_r = np.full(n, -50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for Williams %R and EMA
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_williams_r = williams_r[i]
        curr_ema = ema_50_aligned[i]
        
        # Volume confirmation: volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (2.0 * vol_ma_20)
        else:
            volume_confirm = False
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -90 (extreme oversold), price above 12h EMA50, volume spike
            if (curr_williams_r < -90.0 and 
                curr_close > curr_ema and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Williams %R > -10 (extreme overbought), price below 12h EMA50, volume spike
            elif (curr_williams_r > -10.0 and 
                  curr_close < curr_ema and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -50 (return from oversold) OR price crosses below EMA
            if (curr_williams_r > -50.0 or 
                curr_close < curr_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -50 (return from overbought) OR price crosses above EMA
            if (curr_williams_r < -50.0 or 
                curr_close > curr_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals