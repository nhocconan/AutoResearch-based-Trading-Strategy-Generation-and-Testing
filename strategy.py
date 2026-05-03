#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above R3 + volume spike + price > 1d EMA50 (uptrend).
# Short when price breaks below S3 + volume spike + price < 1d EMA50 (downtrend).
# Uses proven Camarilla structure with tight entries to avoid overtrading.
# Designed for 50-150 total trades over 4 years. Works in both bull (breakout long) and bear (breakdown short).

name = "12h_Camarilla_R3S3_1dEMA50_Trend_VolumeSpike"
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
    open_ = prices['open'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Camarilla levels (R3, S3)
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4, etc.
    # Standard Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Simplified: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low) [using common multiplier]
    # Using actual formula: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Which simplifies to: R3 = close + 1.21*(high-low)/4, S3 = close - 1.21*(high-low)/4
    # Or: R3 = close + 0.3025*(high-low), S3 = close - 0.3025*(high-low)
    # Commonly approximated as: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    # We'll use the standard: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    camarilla_multiplier = 1.1
    r3_1d = df_1d['close'].values + camarilla_multiplier * (df_1d['high'].values - df_1d['low'].values)
    s3_1d = df_1d['close'].values - camarilla_multiplier * (df_1d['high'].values - df_1d['low'].values)
    
    # Align Camarilla levels to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume regime: current 12h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        close_val = close[i]
        open_val = open_[i]
        high_val = high[i]
        low_val = low[i]
        ema_trend = ema_50_1d_aligned[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_trend) or np.isnan(r3) or np.isnan(s3):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine trend regime: bull if close > 1d EMA50, bear if close < 1d EMA50
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Breakout conditions: price breaks above R3 or below S3
        # Using close price for breakout confirmation
        bull_breakout = close_val > r3
        bear_breakout = close_val < s3
        
        # Generate signals
        if position == 0:
            # Long: bull breakout + volume spike + bull trend
            if bull_breakout and vol_spike and is_bull_trend:
                signals[i] = 0.25
                position = 1
            # Short: bear breakout + volume spike + bear trend
            elif bear_breakout and vol_spike and is_bear_trend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on bear breakout or trend change to bear
            if bear_breakout or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on bull breakout or trend change to bull
            if bull_breakout or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals