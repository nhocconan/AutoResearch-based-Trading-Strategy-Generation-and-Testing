#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 12h Camarilla pivot levels (based on previous 12h bar)
    # R3 = close + (high - low) * 1.1
    # S3 = close - (high - low) * 1.1
    # Note: using previous bar's high/low/close to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1
    
    # 12h volume spike (24-period average, 2 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_24[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Camarilla R3 with volume and 1d uptrend
            vol_condition = volume[i] > vol_ma_24[i] * 1.5
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > camarilla_r3[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla S3 with volume and 1d downtrend
            elif close[i] < camarilla_s3[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below Camarilla midpoint or volume drops
            camarilla_mid = (camarilla_r3[i] + camarilla_s3[i]) / 2
            if close[i] < camarilla_mid or volume[i] < vol_ma_24[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above Camarilla midpoint or volume drops
            camarilla_mid = (camarilla_r3[i] + camarilla_s3[i]) / 2
            if close[i] > camarilla_mid or volume[i] < vol_ma_24[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA trend filter and volume confirmation
# - Camarilla levels identify key support/resistance from previous period
# - 1d EMA(34) ensures alignment with daily trend
# - Volume spike (1.5x average) confirms breakout strength
# - Works in bull (buy R3 breakouts in uptrend) and bear (sell S3 breakdowns in downtrend)
# - Position size 0.25 targets 15-25 trades/year, avoiding fee drag
# - Exit at Camarilla midpoint provides logical profit target
# - Designed for 12h timeframe to reduce trade frequency and fee drag
# - Uses proper look-ahead avoidance with previous bar data for Camarilla calculation
# - Complies with MTF rules: 1d data loaded once, aligned with delay