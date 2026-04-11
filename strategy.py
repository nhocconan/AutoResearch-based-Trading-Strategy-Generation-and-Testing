#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with 1d trend filter and volume spike
# - Long: Williams %R(14) crosses above -80 (oversold reversal), volume > 2x 20-period avg, price > 1d EMA(50)
# - Short: Williams %R(14) crosses below -20 (overbought reversal), volume > 2x 20-period avg, price < 1d EMA(50)
# - Exit: Williams %R crosses above -20 for longs, below -80 for shorts
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 25-35 trades/year (100-140 total over 4 years) to stay within fee drag limits
# - Williams %R is effective in both bull and bear markets for identifying reversal points
# - Volume spike confirms institutional participation
# - 1d EMA filter ensures trades align with higher timeframe trend

name = "4h_1d_williamsr_volume_v1"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop for EMA trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute Williams %R(14) on 4h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Pre-compute 4h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(williams_r[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        wr_current = williams_r[i]
        wr_prev = williams_r[i-1]
        
        # Volume confirmation: current volume > 2x 20-period average (strict filter)
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # 1d EMA trend bias
        ema_bias_long = close_price > ema_50_1d_aligned[i]
        ema_bias_short = close_price < ema_50_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long reversal: Williams %R crosses above -80 from below, volume confirmation, long bias
        if wr_prev <= -80 and wr_current > -80 and vol_confirm and ema_bias_long:
            enter_long = True
        
        # Short reversal: Williams %R crosses below -20 from above, volume confirmation, short bias
        if wr_prev >= -20 and wr_current < -20 and vol_confirm and ema_bias_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long when Williams %R crosses above -20 (overbought)
            if wr_prev <= -20 and wr_current > -20:
                exit_long = True
        elif position == -1:
            # Exit short when Williams %R crosses below -80 (oversold)
            if wr_prev >= -80 and wr_current < -80:
                exit_short = True
        
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