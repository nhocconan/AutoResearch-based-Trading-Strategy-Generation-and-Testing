#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume spike
# - Williams %R(14): overbought > -20, oversold < -80
# - Trend filter: price > 1d EMA50 for longs, price < 1d EMA50 for shorts
# - Volume confirmation: current volume > 2.0x 20-period 6h volume average
# - Only trade mean reversion in direction of 1d trend (avoid counter-trend traps)
# - Discrete position sizing: ±0.25 to manage drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee limits
# - Williams %R excels at catching reversals in ranging/bear markets (2025+)
# - 1d EMA50 filter ensures we only take trades aligned with higher timeframe trend
# - Volume spike confirms institutional interest behind the move

name = "6h_1d_williamsr_meanrev_volume_v3"
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
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute 6h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute Williams %R on 6h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume data
        price_current = close[i]
        volume_current = volume[i]
        
        # Williams %R conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # 1d trend filter
        uptrend = price_current > ema50_1d_aligned[i]
        downtrend = price_current < ema50_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_spike = volume_current > 2.0 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: oversold + uptrend + volume spike
        if oversold and uptrend and vol_spike:
            enter_long = True
        
        # Short: overbought + downtrend + volume spike
        if overbought and downtrend and vol_spike:
            enter_short = True
        
        # Exit conditions: reverse Williams %R condition or loss of trend
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if not oversold OR trend turns down
            exit_long = (not oversold) or (not uptrend)
        elif position == -1:
            # Exit short if not overbought OR trend turns up
            exit_short = (not overbought) or (not downtrend)
        
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