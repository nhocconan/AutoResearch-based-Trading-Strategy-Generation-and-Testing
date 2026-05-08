#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d trend filter and volume confirmation
# Williams %R > -20 = overbought, < -80 = oversold
# Long: Williams %R crosses above -80 (exit oversold) + price > 1d EMA50 + volume spike
# Short: Williams %R crosses below -20 (enter overbought) + price < 1d EMA50 + volume spike
# Exit: Williams %R crosses opposite threshold (-20 for long, -80 for short)
# Williams %R captures momentum reversals, EMA50 filters trend direction, volume confirms strength
# Designed for 15-30 trades/year with controlled frequency

name = "6h_WilliamsR_1dEMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 60m data for Williams %R calculation (14-period)
    df_60m = get_htf_data(prices, '60m')
    if len(df_60m) < 14:
        return np.zeros(n)
    
    # Get 1d data for EMA50 trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams %R (14-period)
    high_60m = df_60m['high'].values
    low_60m = df_60m['low'].values
    close_60m = df_60m['close'].values
    
    # Williams %R = -100 * (HHV - Close) / (HHV - LLV)
    highest_high = pd.Series(high_60m).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_60m).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_60m) / (highest_high - lowest_low)
    # Handle division by zero when HHV == LLV
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-day average volume for volume filter
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_60m, williams_r)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14)  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 1d volume > 1.5x 20-day average
        # Find the most recent completed 1d bar
        idx_1d = len(df_1d) - 1
        while idx_1d >= 0 and df_1d.iloc[idx_1d]['open_time'] > prices.iloc[i]['open_time']:
            idx_1d -= 1
        vol_filter = False
        if idx_1d >= 0:
            vol_1d_current = df_1d.iloc[idx_1d]['volume']
            vol_filter = vol_1d_current > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for mean reversion entries with trend and volume confirmation
            # Long: Williams %R crosses above -80 (exit oversold) + uptrend + volume spike
            if williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and ema50_1d_aligned[i] > 0:
                if vol_filter:
                    signals[i] = 0.25
                    position = 1
            # Short: Williams %R crosses below -20 (enter overbought) + downtrend + volume spike
            elif williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and ema50_1d_aligned[i] < 0:
                if vol_filter:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Williams %R crosses below -20 (overbought threshold)
            if williams_r_aligned[i] < -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses above -80 (oversold threshold)
            if williams_r_aligned[i] > -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals