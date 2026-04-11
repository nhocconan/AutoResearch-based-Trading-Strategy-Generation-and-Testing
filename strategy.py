#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d trend filter and volume spike confirmation
# - Williams %R(14) identifies overbought/oversold conditions
# - Long: %R crosses above -80 from below + price > 1d EMA50 (uptrend) + volume > 2x 20-period average
# - Short: %R crosses below -20 from above + price < 1d EMA50 (downtrend) + volume > 2x 20-period average
# - Exit: Opposite %R crossover or Donchian(10) breakout in opposite direction
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Williams %R is effective at catching reversals in ranging markets (common in 2025 BTC/ETH)
# - 1d EMA50 filter ensures we trade with the higher timeframe trend
# - Volume spike confirmation filters out weak reversals and increases signal quality

name = "6h_1d_williamsr_reversal_v1"
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
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute Williams %R on 6h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Pre-compute Donchian(10) for exit signals
    highest_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lowest_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(highest_high_10[i]) or 
            np.isnan(lowest_low_10[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Williams %R values
        wr_current = williams_r[i]
        wr_previous = williams_r[i-1]
        
        # Trend filter: price vs 1d EMA50
        uptrend = close_price > ema_50_1d_aligned[i]
        downtrend = close_price < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 2x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20_1d_aligned[i]
        
        # Donchian levels for exit
        upper_channel_10 = highest_high_10[i]
        lower_channel_10 = lowest_low_10[i]
        
        # Williams %R crossover signals
        wr_cross_above_80 = (wr_previous <= -80) and (wr_current > -80)
        wr_cross_below_20 = (wr_previous >= -20) and (wr_current < -20)
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long reversal: Williams %R crosses above -80 from below + uptrend + volume confirmation
        if wr_cross_above_80 and uptrend and vol_confirm:
            enter_long = True
        
        # Short reversal: Williams %R crosses below -20 from above + downtrend + volume confirmation
        if wr_cross_below_20 and downtrend and vol_confirm:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Williams %R crosses below -20 or price breaks below Donchian low
            exit_long = wr_cross_below_20 or (close_price < lower_channel_10)
        elif position == -1:
            # Exit short if Williams %R crosses above -80 or price breaks above Donchian high
            exit_short = wr_cross_above_80 or (close_price > upper_channel_10)
        
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