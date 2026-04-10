#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume spike confirmation
# - Long when Williams %R(14) < -80 (oversold) AND 1d close > 1d EMA50 (bullish trend) AND volume spike (>2x avg)
# - Short when Williams %R(14) > -20 (overbought) AND 1d close < 1d EMA50 (bearish trend) AND volume spike (>2x avg)
# - Exit when Williams %R returns to neutral range (-50) or opposite extreme reached
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 12-37 trades/year on 6h timeframe to stay within fee drag limits
# - Williams %R is effective in ranging markets and catches reversals in bear market rallies

name = "6h_1d_williamsr_meanreversion_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Williams %R(14) on 6h
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d close for trend comparison
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate 6h volume average for spike confirmation
    volume_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # ~4 days average
    
    for i in range(50, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(close_1d_aligned[i]) or np.isnan(volume_avg[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike confirmation: current volume > 2x 24-period average
        vol_spike = volume[i] > 2.0 * volume_avg[i]
        
        # Trend filter: 1d close vs 1d EMA50
        trend_bullish = close_1d_aligned[i] > ema_50_1d_aligned[i]
        trend_bearish = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Williams %R levels
        wr = williams_r[i]
        oversold = wr < -80
        overbought = wr > -20
        neutral_exit = wr > -50 and wr < -50  # This will never be true, fixing below
        
        # Fixed neutral exit condition
        neutral_exit = (wr >= -50)  # Exit when Williams %R returns to or above -50
        
        # Exit conditions
        exit_long = neutral_exit or overbought  # Exit long when neutral or overbought
        exit_short = neutral_exit or oversold   # Exit short when neutral or oversold
        
        if position == 0:  # Flat - look for entry
            if oversold and trend_bullish and vol_spike:
                position = 1
                signals[i] = 0.25
            elif overbought and trend_bearish and vol_spike:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals