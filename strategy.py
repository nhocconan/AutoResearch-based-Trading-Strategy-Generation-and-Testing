#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d RSI extremes with 1w Supertrend trend filter and volume spike confirmation
# Long when 1d RSI < 30 (oversold) AND 1w Supertrend = uptrend AND volume > 2.0 * avg_volume(20) on 12h
# Short when 1d RSI > 70 (overbought) AND 1w Supertrend = downtrend AND volume > 2.0 * avg_volume(20) on 12h
# Exit when 1d RSI crosses back through 50 (mean reversion to midpoint)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# RSI extremes provide high-probability reversal points in ranging markets
# 1w Supertrend filter ensures we trade with the dominant weekly trend
# Volume spike confirmation (2.0x) validates reversal strength while limiting overtrading
# Works in both bull (buy oversold dips) and bear (sell overbought rallies) markets

name = "12h_1dRSI_Extreme_1wSupertrend_Trend_VolumeSpike"
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
    
    # Get 1d data ONCE before loop for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need at least 14 completed 1d bars for RSI
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.fillna(50).values  # Fill NaN with 50 (neutral)
    
    # Align 1d RSI to 12h timeframe (wait for completed 1d bar)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Get 1w data ONCE before loop for Supertrend trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:  # Need at least 10 completed weekly bars for Supertrend
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Supertrend (ATR=10, multiplier=3.0)
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = pd.Series(high_1w) - pd.Series(low_1w)
    tr2 = abs(pd.Series(high_1w) - pd.Series(close_1w).shift(1))
    tr3 = abs(pd.Series(low_1w) - pd.Series(close_1w).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (pd.Series(high_1w) + pd.Series(low_1w)) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.zeros(len(close_1w))
    direction = np.ones(len(close_1w))  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_1w)):
        if close_1w[i] > upper_band[i-1]:
            direction[i] = 1
        elif close_1w[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
    
        supertrend[i] = (lower_band[i] if direction[i] == 1 else upper_band[i])
    
    # Align 1w Supertrend direction to 12h timeframe (wait for completed 1w bar)
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_1w, direction)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(rsi_aligned[i]) or np.isnan(supertrend_direction_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold), Supertrend uptrend (direction=1), volume spike, in session
            if (rsi_aligned[i] < 30 and 
                supertrend_direction_aligned[i] == 1 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought), Supertrend downtrend (direction=-1), volume spike, in session
            elif (rsi_aligned[i] > 70 and 
                  supertrend_direction_aligned[i] == -1 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI crosses back above 50 (mean reversion)
            if rsi_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI crosses back below 50 (mean reversion)
            if rsi_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals