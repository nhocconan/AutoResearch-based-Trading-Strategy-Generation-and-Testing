#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R overbought/oversold + 1w trend filter + volume confirmation
# - Williams %R(14) from 1w: identifies overbought (> -20) and oversold (< -80) conditions
# - Long when %R crosses above -80 from below (oversold bounce) with 1w uptrend (price > EMA50)
# - Short when %R crosses below -20 from above (overbought rejection) with 1w downtrend (price < EMA50)
# - Volume confirmation: current volume > 1.5x 20-period average to avoid low-volume false signals
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits for 12h
# - Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets
# - 1w HTF provides reliable trend context, reducing counter-trend whipsaws

name = "12h_1w_williamsr_trend_volume_v1"
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
    
    # Load 1w data ONCE before loop for Williams %R, trend, and volume
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Extract 1w arrays
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1w) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1w volume SMA (20-period)
    volume_sma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align 1w indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1w, volume_sma_20_1w)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        volume_current = volume[i]
        
        # Williams %R signals
        wr_prev = williams_r_aligned[i-1] if i > 0 else williams_r_aligned[i]
        wr_curr = williams_r_aligned[i]
        
        # Oversold bounce: %R crosses above -80 from below
        oversold_bounce = (wr_prev <= -80) and (wr_curr > -80)
        
        # Overbought rejection: %R crosses below -20 from above
        overbought_rejection = (wr_prev >= -20) and (wr_curr < -20)
        
        # Trend filter: price above/below EMA50
        uptrend = price_close > ema_50_aligned[i]
        downtrend = price_close < ema_50_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Oversold bounce + uptrend + volume confirmation
        if oversold_bounce and uptrend and vol_confirm:
            enter_long = True
        
        # Short: Overbought rejection + downtrend + volume confirmation
        if overbought_rejection and downtrend and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite Williams %R cross or trend change
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if overbought rejection occurs OR trend turns down
            exit_long = overbought_rejection or (not uptrend)
        elif position == -1:
            # Exit short if oversold bounce occurs OR trend turns up
            exit_short = oversold_bounce or (not downtrend)
        
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