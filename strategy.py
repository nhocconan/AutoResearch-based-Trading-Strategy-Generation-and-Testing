#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Williams %R(14) from 6h: oversold < -80, overbought > -20
# - 1d EMA(50) trend filter: long only when price > EMA50, short only when price < EMA50
# - Volume confirmation: current 6h volume > 1.5x 20-period 6h volume SMA
# - Discrete position sizing: ±0.25 to manage drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within 6h fee drag limits
# - Williams %R excels at catching reversals in both bull rallies and bear traps
# - 1d EMA filter ensures we trade with the higher timeframe trend, reducing whipsaw
# - Volume confirmation adds conviction to mean reversion signals

name = "6h_1d_williamsr_volume_trend_v1"
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
    
    # Pre-compute 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Pre-compute 6h volume SMA (20-period)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA(50)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_sma_20[i]) or 
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        volume_current = volume[i]
        
        # Williams %R conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # 1d EMA trend filter
        uptrend = price_close > ema_50_aligned[i]
        downtrend = price_close < ema_50_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Williams %R oversold + volume confirmation + 1d uptrend
        if oversold and vol_confirm and uptrend:
            enter_long = True
        
        # Short: Williams %R overbought + volume confirmation + 1d downtrend
        if overbought and vol_confirm and downtrend:
            enter_short = True
        
        # Exit conditions: opposite Williams %R extreme or volume dry-up
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Williams %R becomes overbought OR volume confirmation fails
            exit_long = williams_r[i] > -20 or not vol_confirm
        elif position == -1:
            # Exit short if Williams %R becomes oversold OR volume confirmation fails
            exit_short = williams_r[i] < -80 or not vol_confirm
        
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