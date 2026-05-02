#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla levels provide precise intraday support/resistance; breakout above R3 or below S3 with volume spike captures strong momentum
# 1d EMA34 filters for primary trend alignment (price > EMA for longs, < EMA for shorts)
# Volume spike (2.5x 20-period average) ensures institutional participation and reduces false breakouts
# Uses discrete position sizing 0.25 to minimize fee churn
# Targets 12-25 trades/year (50-100 total over 4 years) to stay within fee drag limits for 12h timeframe
# Works in both bull and bear markets by requiring volume confirmation and primary trend alignment

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla levels from previous 1d bar (using typical price)
    typical_price = (high + low + close) / 3
    # Use previous day's typical price for Camarilla calculation (shifted by 1)
    prev_typical = pd.Series(typical_price).shift(1).values
    # Camarilla levels based on previous day's range
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    # But we need to calculate from previous day's data
    # Load 1d OHLC for Camarilla calculation
    if len(df_1d) < 2:
        return np.zeros(n)
    # Previous 1d close, high, low
    prev_1d_close = df_1d['close'].iloc[-2] if len(df_1d) >= 2 else df_1d['close'].iloc[-1]
    prev_1d_high = df_1d['high'].iloc[-2] if len(df_1d) >= 2 else df_1d['high'].iloc[-1]
    prev_1d_low = df_1d['low'].iloc[-2] if len(df_1d) >= 2 else df_1d['low'].iloc[-1]
    # Calculate Camarilla levels for current 12h bar based on previous 1d
    camarilla_r3 = prev_1d_close + (prev_1d_high - prev_1d_low) * 1.1 / 4
    camarilla_s3 = prev_1d_close - (prev_1d_high - prev_1d_low) * 1.1 / 4
    # Broadcast to all 12h bars (levels remain constant until new 1d bar)
    camarilla_r3_arr = np.full(n, camarilla_r3)
    camarilla_s3_arr = np.full(n, camarilla_s3)
    
    # Calculate volume spike (2.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for volume MA and EMA)
    start_idx = 50  # max(20 for volume, 34 for EMA) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(camarilla_r3_arr[i]) or np.isnan(camarilla_s3_arr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3 + price > 1d EMA + volume spike
            if close[i] > camarilla_r3_arr[i] and close[i] > ema_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 + price < 1d EMA + volume spike
            elif close[i] < camarilla_s3_arr[i] and close[i] < ema_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price retreats to Camarilla S3 level (strong support)
            if close[i] < camarilla_s3_arr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises to Camarilla R3 level (strong resistance)
            if close[i] > camarilla_r3_arr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals