#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and 1w EMA trend filter
# Camarilla pivot levels provide precise intraday support/resistance; R3/S3 are strong reversal levels
# Volume spike (2.0x 20-period average) confirms institutional participation at key levels
# 1w EMA filter ensures we only trade in the direction of the weekly trend to avoid counter-trend whipsaws
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 19-50 trades/year (75-200 total over 4 years) to stay within fee drag limits

name = "4h_Camarilla_R3S3_Breakout_1dVolumeSpike_1wEMATrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume confirmation (2.0x 20-period average)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike_1d = df_1d['volume'].values > (vol_ma_1d * 2.0)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate 1w EMA for trend filter (34-period)
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 4h Camarilla pivot levels (based on previous day's OHLC)
    # We need to align daily OHLC to 4h bars
    # First get daily OHLC from 1d data
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R3 = Close + (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/4
    camarilla_r3 = daily_close + (daily_high - daily_low) * 1.1 / 4
    camarilla_s3 = daily_close - (daily_high - daily_low) * 1.1 / 4
    
    # Align daily Camarilla levels to 4h bars
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Camarilla calculation and EMA)
    start_idx = 50  # buffer for indicators
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 1w EMA
        uptrend = close[i] > ema_34_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only take trades in direction of weekly trend
            if uptrend:
                # In uptrend: look for long at S3 support
                if close[i] <= camarilla_s3_aligned[i] and volume_spike_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            else:
                # In downtrend: look for short at R3 resistance
                if close[i] >= camarilla_r3_aligned[i] and volume_spike_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions: price reaches Camarilla R3 or closes below S3
            if close[i] >= camarilla_r3_aligned[i] or close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price reaches Camarilla S3 or closes above R3
            if close[i] <= camarilla_s3_aligned[i] or close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals