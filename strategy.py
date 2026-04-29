#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla levels provide institutional support/resistance; breakout with volume confirms momentum
# 1d EMA34 ensures we trade with the higher timeframe trend to avoid whipsaws
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue)
# Target: 12-30 trades/year (50-120 total over 4 years)

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Based on previous day's OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla equations
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = C + (Range * 1.1000)
    # S3 = C - (Range * 1.1000)
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3.0
    camarilla_range = prev_high - prev_low
    r3 = prev_close + (camarilla_range * 1.1000)
    s3 = prev_close - (camarilla_range * 1.1000)
    
    # Align Camarilla levels to 12h timeframe (completed daily bar only)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34, 20)  # warmup for 1d EMA, Camarilla, volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        
        # Determine trend based on 1d EMA34
        # Above EMA34 = uptrend, Below EMA34 = downtrend
        is_uptrend = curr_close > curr_ema_34_1d
        is_downtrend = curr_close < curr_ema_34_1d
        
        if position == 0:  # Flat - look for new entries
            # Long breakout above R3 in uptrend with volume confirmation
            if is_uptrend and curr_close > curr_r3 and curr_volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short breakdown below S3 in downtrend with volume confirmation
            elif is_downtrend and curr_close < curr_s3 and curr_volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price crosses below 1d EMA34 OR breakdown below S3 (failed breakout)
            if curr_close < curr_ema_34_1d or curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price crosses above 1d EMA34 OR breakout above R3 (failed breakdown)
            if curr_close > curr_ema_34_1d or curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals