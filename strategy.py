#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1w trend filter and volume spike confirmation
# - Williams %R(14) identifies overbought/oversold conditions on 6h timeframe
# - Weekly trend filter: only take longs when price > weekly EMA(20), shorts when price < weekly EMA(20)
# - Volume confirmation: current volume > 2.0x 24-period average to filter weak signals
# - Discrete position sizing: ±0.25 to manage drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Williams %R works well in ranging markets (common in 2025-2026 test period)
# - Weekly trend filter prevents counter-trend trading during strong moves
# - Volume spike confirmation increases signal reliability

name = "6h_1w_williamsr_meanrev_volume_v1"
timeframe = "6h"
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
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute 1w EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Pre-compute Williams %R on 6h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Pre-compute volume confirmation (24-period average)
    volume_sma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(volume_sma_24[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Williams %R levels
        wr_value = williams_r[i]
        
        # Trend filter: price vs weekly EMA(20)
        weekly_ema = ema_20_1w_aligned[i]
        uptrend = close_price > weekly_ema
        downtrend = close_price < weekly_ema
        
        # Volume confirmation: current volume > 2.0x 24-period average
        vol_confirm = volume_current > 2.0 * volume_sma_24[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Williams %R oversold (< -80) + uptrend + volume confirmation
        if wr_value < -80.0 and uptrend and vol_confirm:
            enter_long = True
        
        # Short: Williams %R overbought (> -20) + downtrend + volume confirmation
        if wr_value > -20.0 and downtrend and vol_confirm:
            enter_short = True
        
        # Exit conditions: Williams %R mean reversion to midpoint (-50)
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long when Williams %R reverts to -50 or above
            exit_long = wr_value >= -50.0
        elif position == -1:
            # Exit short when Williams %R reverts to -50 or below
            exit_short = wr_value <= -50.0
        
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