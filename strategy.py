#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend direction with 1d volume spike and 1d Bollinger Band squeeze filter
# - Long: KAMA(10,2,30) rising, volume > 2.0x 20-period average, BB Width < 20th percentile (low volatility squeeze)
# - Short: KAMA falling, volume > 2.0x 20-period average, BB Width < 20th percentile
# - Exit: KAMA direction reverses or ATR-based stop (2.0 ATR)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - KAMA adapts to market noise, effective in both trending and ranging markets
# - Volume spike confirms institutional participation
# - Bollinger Band squeeze identifies low volatility periods preceding breakouts

name = "12h_1d_kama_volume_bbwidth_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 1d data ONCE before loop for KAMA, volume, and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d KAMA(10,2,30)
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, 10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close_1d, 1)), axis=0)  # 10-period sum of absolute changes
    # Pad the beginning with zeros for alignment
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing Constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # Start with first close
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align 1d KAMA to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Pre-compute 1d KAMA previous value for direction detection
    kama_prev = np.roll(kama_aligned, 1)
    kama_prev[0] = kama_aligned[0]
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 1d Bollinger Band Width (20,2)
    bb_middle = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle
    # Handle division by zero
    bb_width = np.where(bb_middle == 0, np.nan, bb_width)
    
    # Align 1d BB Width to 12h timeframe
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # Calculate 20th percentile of BB Width for squeeze filter (using expanding window)
    bb_width_percentile_20 = np.full_like(bb_width_aligned, np.nan)
    for i in range(len(bb_width_aligned)):
        if i >= 20:  # Need at least 20 values for percentile
            valid_values = bb_width_aligned[max(0, i-100):i+1]  # Use last 100 values or available
            valid_values = valid_values[~np.isnan(valid_values)]
            if len(valid_values) >= 20:
                bb_width_percentile_20[i] = np.percentile(valid_values, 20)
    
    # Pre-compute ATR for stoploss (12h timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(kama_prev[i]) or np.isnan(volume_sma_20_aligned[i]) or
            np.isnan(bb_width_aligned[i]) or np.isnan(bb_width_percentile_20[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # KAMA values
        kama_current = kama_aligned[i]
        kama_previous = kama_prev[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20_aligned[i]
        
        # Bollinger Band squeeze filter: BB Width < 20th percentile (low volatility)
        bb_squeeze = bb_width_aligned[i] < bb_width_percentile_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: KAMA rising + volume spike + BB squeeze
        if kama_current > kama_previous and vol_confirm and bb_squeeze:
            enter_long = True
        
        # Short: KAMA falling + volume spike + BB squeeze
        if kama_current < kama_previous and vol_confirm and bb_squeeze:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if KAMA reverses or ATR-based stop
            exit_long = (kama_current < kama_previous) or (close_price <= entry_price - 2.0 * atr_14[i])
        elif position == -1:
            # Exit short if KAMA reverses or ATR-based stop
            exit_short = (kama_current > kama_previous) or (close_price >= entry_price + 2.0 * atr_14[i])
        
        # Track entry price for stoploss calculation
        if enter_long or enter_short:
            entry_price = close_price
        
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