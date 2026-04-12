#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1d_camarilla_pivot_reversal_v1
# Use daily Camarilla pivot levels (support/resistance) on 12h timeframe.
# Enter long at S3 with bullish price action, short at R3 with bearish price action.
# Filter by 1d EMA21 trend to avoid counter-trend traps in strong trends.
# Low frequency expected due to specific price levels + trend filter.
name = "12h_1d_camarilla_pivot_reversal_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_ = prices['open'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = C + ((H-L)*1.1/2)
    # R3 = C + ((H-L)*1.1/4)
    # R2 = C + ((H-L)*1.1/6)
    # R1 = C + ((H-L)*1.1/12)
    # S1 = C - ((H-L)*1.1/12)
    # S2 = C - ((H-L)*1.1/6)
    # S3 = C - ((H-L)*1.1/4)
    # S4 = C - ((H-L)*1.1/2)
    # We'll use R3 and S3 as primary entry levels
    range_1d = high_1d - low_1d
    camarilla_r3 = close_1d + (range_1d * 1.1 / 4)
    camarilla_s3 = close_1d - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d EMA21 for trend filter
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if indicators not ready
        if np.isnan(ema_21_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions with trend filter
        # Long: price at or below S3 AND bullish candle close in uptrend
        bullish_setup = (
            close[i] <= camarilla_s3_aligned[i] and  # at or below S3 support
            close[i] > open_[i] and  # bullish candle
            close[i] > ema_21_aligned[i]  # above EMA21 (uptrend filter)
        )
        
        # Short: price at or above R3 AND bearish candle close in downtrend
        bearish_setup = (
            close[i] >= camarilla_r3_aligned[i] and  # at or above R3 resistance
            close[i] < open_[i] and  # bearish candle
            close[i] < ema_21_aligned[i]  # below EMA21 (downtrend filter)
        )
        
        # Exit conditions: opposite Camarilla level touch or trend reversal
        exit_long = (
            close[i] >= camarilla_r3_aligned[i] or  # hit R3 resistance
            close[i] < ema_21_aligned[i]  # trend turned down
        )
        
        exit_short = (
            close[i] <= camarilla_s3_aligned[i] or  # hit S3 support
            close[i] > ema_21_aligned[i]  # trend turned up
        )
        
        if bullish_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals