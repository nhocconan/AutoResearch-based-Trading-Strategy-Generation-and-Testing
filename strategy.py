#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels from 1d: long at S3 bounce with volume confirmation, short at R3 rejection with volume confirmation
# - Long: price touches/bounces above S3 with volume > 1.5x 20-period average and RSI(14) < 40 (oversold)
# - Short: price touches/rejects below R3 with volume > 1.5x 20-period average and RSI(14) > 60 (overbought)
# - Exit: price reverts to midpoint between S3 and R3 (mean reversion)
# - Uses 1d Camarilla levels calculated from prior 1d OHLC, aligned to 4h
# - Works in both bull and bear markets by fading extremes at Camarilla levels
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits

name = "4h_1d_camarilla_pivot_fade_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Camarilla levels (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Pre-compute 1d Camarilla levels (based on prior day OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4
    range_1d = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * range_1d * 1.1 / 4
    camarilla_s3 = close_1d - 1.1 * range_1d * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (use prior day's levels for current day)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Pre-compute 4h volume confirmation (20-period average)
    volume = prices['volume'].values
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute RSI(14) on 4h close
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        rsi_current = rsi_values[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Price position relative to Camarilla levels
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        midpoint = (r3 + s3) / 2
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: price at/below S3 with volume confirmation and oversold RSI
        if close_price <= s3 * 1.001 and vol_confirm and rsi_current < 40:
            enter_long = True
        
        # Short: price at/above R3 with volume confirmation and overbought RSI
        if close_price >= r3 * 0.999 and vol_confirm and rsi_current > 60:
            enter_short = True
        
        # Exit conditions: mean reversion to midpoint
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price reaches midpoint
            exit_long = close_price >= midpoint
        elif position == -1:
            # Exit short if price reaches midpoint
            exit_short = close_price <= midpoint
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals