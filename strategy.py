#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Uses discrete sizing 0.25 to minimize fee churn. Designed to work in both bull (breakouts) and bear (breakdowns).
# Camarilla levels provide structured support/resistance; EMA34 filters trend direction; volume confirms breakout strength.
# Target: 20-50 trades/year per symbol to avoid fee drag while capturing meaningful moves.

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Load 4h data ONCE before loop for volume MA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA34
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
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
        
        # Skip if any data not ready
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_vol_ma = vol_ma_4h_aligned[i]
        
        # Calculate Camarilla levels for R3 and S3 using prior 4h bar
        # Camarilla: based on prior bar's range
        if i == 0:
            signals[i] = 0.0
            continue
        prior_close = close[i-1]
        prior_high = high[i-1]
        prior_low = low[i-1]
        prior_range = prior_high - prior_low
        
        if prior_range <= 0:
            signals[i] = 0.0
            continue
            
        camarilla_r3 = prior_close + (prior_range * 1.1 / 4)
        camarilla_s3 = prior_close - (prior_range * 1.1 / 4)
        
        # Volume confirmation: current 4h volume > 2.0x 20-period average
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: break above R3 with volume and price > 1d EMA34 (uptrend)
            if (curr_close > camarilla_r3 and 
                volume_confirm and 
                curr_close > curr_ema_34_1d):
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with volume and price < 1d EMA34 (downtrend)
            elif (curr_close < camarilla_s3 and 
                  volume_confirm and 
                  curr_close < curr_ema_34_1d):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: close below S3 (breakdown) or price < 1d EMA34 (trend violation)
            if (curr_close < camarilla_s3 or 
                curr_close < curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: close above R3 (breakout) or price > 1d EMA34 (trend violation)
            if (curr_close > camarilla_r3 or 
                curr_close > curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals