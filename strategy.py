#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1d trend filter and volume spike
# - Williams %R(14) < -80 (oversold) + price > 1d EMA(50) + volume > 1.5x 20-period avg → long
# - Williams %R(14) > -20 (overbought) + price < 1d EMA(50) + volume > 1.5x 20-period avg → short
# - Exit: Williams %R crosses above -50 (for long) or below -50 (for short)
# - Uses 6h timeframe to reduce trade frequency vs lower TFs
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-30 trades/year (50-120 total over 4 years) to stay within fee drag limits
# - Williams %R captures mean reversion in extreme conditions, effective in both bull and bear markets

name = "6h_1d_williamsr_extreme_v1"
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
    
    # Load 1d data ONCE before loop for EMA trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 6h Williams %R(14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Pre-compute 6h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Williams %R levels
        wr = williams_r[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # 1d EMA trend bias
        ema_bias_long = close_price > ema_50_1d_aligned[i]
        ema_bias_short = close_price < ema_50_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long reversal: Williams %R deeply oversold, volume spike, long bias
        if wr < -80 and vol_confirm and ema_bias_long:
            enter_long = True
        
        # Short reversal: Williams %R deeply overbought, volume spike, short bias
        if wr > -20 and vol_confirm and ema_bias_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long when Williams %R crosses above -50 (reversal losing momentum)
            exit_long = wr > -50
        elif position == -1:
            # Exit short when Williams %R crosses below -50 (reversal losing momentum)
            exit_short = wr < -50
        
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