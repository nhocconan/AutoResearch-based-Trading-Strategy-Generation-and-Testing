#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d trend filter and volume spike confirmation
# - Long: Williams %R(14) crosses above -80 (oversold) + price > 1d EMA50 (uptrend) + volume > 2.0x 20-period average
# - Short: Williams %R(14) crosses below -20 (overbought) + price < 1d EMA50 (downtrend) + volume > 2.0x 20-period average
# - Exit: Opposite Williams %R crossing (%R < -80 for long exit, %R > -20 for short exit) or volume drop below average
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Williams %R identifies exhaustion points in both bull and bear markets
# - 1d EMA50 filter ensures we trade with the higher timeframe trend
# - Volume spike confirmation filters out false reversals and increases signal quality

name = "6h_1d_williamsr_reversal_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return signals
    
    # Pre-compute 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 1d volume average for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute Williams %R on 6h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        williams_r_current = williams_r[i]
        williams_r_prev = williams_r[i-1]
        
        # Trend filter: price vs 1d EMA50
        uptrend = close_price > ema_50_1d_aligned[i]
        downtrend = close_price < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20_aligned[i]
        
        # Williams %R crossovers
        cross_above_80 = (williams_r_prev <= -80) and (williams_r_current > -80)
        cross_below_20 = (williams_r_prev >= -20) and (williams_r_current < -20)
        cross_below_80 = (williams_r_prev >= -80) and (williams_r_current < -80)
        cross_above_20 = (williams_r_prev <= -20) and (williams_r_current > -20)
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Williams %R crosses above -80 (oversold) + uptrend + volume confirmation
        if cross_above_80 and uptrend and vol_confirm:
            enter_long = True
        
        # Short: Williams %R crosses below -20 (overbought) + downtrend + volume confirmation
        if cross_below_20 and downtrend and vol_confirm:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Williams %R crosses below -80 or volume drops below average
            exit_long = cross_below_80 or (volume_current < volume_sma_20_aligned[i])
        elif position == -1:
            # Exit short if Williams %R crosses above -20 or volume drops below average
            exit_short = cross_above_20 or (volume_current < volume_sma_20_aligned[i])
        
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