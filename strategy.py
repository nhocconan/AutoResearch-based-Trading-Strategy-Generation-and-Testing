#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA(34) trend filter and volume confirmation
# Long when price breaks above Camarilla R3 (1d) + volume spike + price > 1d EMA(34)
# Short when price breaks below Camarilla S3 (1d) + volume spike + price < 1d EMA(34)
# Uses Camarilla levels from previous 1d bar to avoid look-ahead
# 1d EMA(34) filter captures intermediate trend and reduces whipsaw
# Volume spike (2.0x 20-period average) confirms institutional participation
# Designed for low trade frequency (12-37/year on 12h) to minimize fee drag
# Works in both bull (breakouts) and bear (mean reversion at extremes) markets

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and EMA(34) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 12h timeframe (wait for completed 1d bar)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels (R3, S3) from previous 1d bar
    # Based on previous day's high, low, close
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous bar's high, low, close to calculate today's Camarilla levels
        high_prev = df_1d['high'].iloc[i-1] if i-1 < len(df_1d) else np.nan
        low_prev = df_1d['low'].iloc[i-1] if i-1 < len(df_1d) else np.nan
        close_prev = df_1d['close'].iloc[i-1] if i-1 < len(df_1d) else np.nan
        
        if not (np.isnan(high_prev) or np.isnan(low_prev) or np.isnan(close_prev)):
            # Camarilla formula: R3 = Close + (High - Low) * 1.1/4
            #                S3 = Close - (High - Low) * 1.1/4
            camarilla_r3[i] = close_prev + (high_prev - low_prev) * 1.1 / 4
            camarilla_s3[i] = close_prev - (high_prev - low_prev) * 1.1 / 4
    
    # Volume confirmation (2.0x 20-period average) on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 50  # max(1 for Camarilla, 20 for volume MA, 34 for 1d EMA)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 + volume spike + price > 1d EMA(34)
            if (close[i] > camarilla_r3[i] and volume_spike[i] and close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Camarilla S3 + volume spike + price < 1d EMA(34)
            elif (close[i] < camarilla_s3[i] and volume_spike[i] and close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Camarilla S3 OR price below 1d EMA(34)
            if (close[i] < camarilla_s3[i] or close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla R3 OR price above 1d EMA(34)
            if (close[i] > camarilla_r3[i] or close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals