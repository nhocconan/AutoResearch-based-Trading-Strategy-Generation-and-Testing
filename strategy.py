#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses Camarilla pivot levels from prior completed 4h bar for structure (breakout = momentum)
# 4h EMA50 filter ensures we trade in direction of higher timeframe trend (avoids counter-trend whipsaws)
# Volume confirmation ensures breakout has sufficient participation (>1.5x 20 EMA volume)
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods
# Discrete sizing 0.20 balances risk and return while minimizing fee churn
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe
# Works in both bull (breakout continuation) and bear (breakdown continuation) markets
# Focus on BTC/ETH by requiring 4h trend alignment (avoids SOL-only bias, more robust across regimes)

name = "1h_Camarilla_R3S3_4hEMA50_VolumeConfirm_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA trend filter and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need enough data for EMA calculation
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_series = pd.Series(close_4h)
    
    # Calculate 4h EMA(50) trend filter from prior completed 4h bar
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_shifted = np.roll(ema_50_4h, 1)
    ema_50_4h_shifted[0] = np.nan
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h_shifted)
    
    # Calculate Camarilla pivot levels from prior completed 4h bar
    # Typical price = (high + low + close) / 3
    typical_price_4h = (high_4h + low_4h + close_4h) / 3.0
    typical_price_series = pd.Series(typical_price_4h)
    pivot_4h = typical_price_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate volatility range for Camarilla levels
    hl_range_4h = high_4h - low_4h
    hl_range_series = pd.Series(hl_range_4h)
    avg_range_4h = hl_range_series.rolling(window=20, min_periods=20).mean().values
    
    # Camarilla R3 and S3 levels (most significant for breakouts)
    # R3 = pivot + (high - low) * 1.1/4
    # S3 = pivot - (high - low) * 1.1/4
    camarilla_upper = pivot_4h + (avg_range_4h * 1.1 / 4.0)
    camarilla_lower = pivot_4h - (avg_range_4h * 1.1 / 4.0)
    
    # Shift by 1 to use only prior completed 4h bar (no look-ahead)
    camarilla_upper_shifted = np.roll(camarilla_upper, 1)
    camarilla_lower_shifted = np.roll(camarilla_lower, 1)
    camarilla_upper_shifted[0] = np.nan
    camarilla_lower_shifted[0] = np.nan
    
    # Align Camarilla levels to 1h timeframe
    camarilla_upper_aligned = align_htf_to_ltf(prices, df_4h, camarilla_upper_shifted)
    camarilla_lower_aligned = align_htf_to_ltf(prices, df_4h, camarilla_lower_shifted)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h_shifted)  # Re-use for clarity
    
    # Volume confirmation: 20-period EMA of volume on 1h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_upper_aligned[i]) or 
            np.isnan(camarilla_lower_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND price > 4h EMA50 AND volume spike
            if close[i] > camarilla_upper_aligned[i] and close[i] > ema_50_4h_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND price < 4h EMA50 AND volume spike
            elif close[i] < camarilla_lower_aligned[i] and close[i] < ema_50_4h_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla S3 OR price crosses below 4h EMA50
            if close[i] < camarilla_lower_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to Camarilla R3 OR price crosses above 4h EMA50
            if close[i] > camarilla_upper_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals