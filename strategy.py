#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d trend filter (EMA34) and volume confirmation
# Uses Camarilla pivot levels calculated from prior 1d bar for structure-based entries
# R3/S3 levels act as strong support/resistance - breaks indicate momentum continuation
# 1d EMA34 ensures trades align with daily trend to avoid counter-trend whipsaws
# Volume spike (2.0x 20-period avg) confirms institutional participation
# Discrete sizing 0.25 balances profit potential with fee drag (target 50-150 trades/4 years)
# Works in bull/bear by only taking breaks in direction of 1d trend

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from prior 1d bar
    # Typical Price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Camarilla levels based on prior day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d
    
    # Camarilla R3, S3, R4, S4 levels
    camarilla_r3 = close_1d + (range_1d * 1.1 / 4)
    camarilla_s3 = close_1d - (range_1d * 1.1 / 4)
    camarilla_r4 = close_1d + (range_1d * 1.1 / 2)
    camarilla_s4 = close_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (completed 1d bar only)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d_series = pd.Series(df_1d['close'])
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Camarilla R3 breakout long: close > R3 with 1d uptrend
            breakout_long = close[i] > camarilla_r3_aligned[i]
            # Camarilla S3 breakdown short: close < S3 with 1d downtrend
            breakout_short = close[i] < camarilla_s3_aligned[i]
            
            # 1d EMA34 trend filter
            ema_long = close[i] > ema_34_1d_aligned[i]
            ema_short = close[i] < ema_34_1d_aligned[i]
            
            if breakout_long and ema_long and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif breakout_short and ema_short and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Camarilla S3 breakdown or trend reversal
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Camarilla R3 breakout or trend reversal
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals