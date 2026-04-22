#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 12h EMA50 trend filter and volume confirmation
# Long when Williams %R crosses above -80 from oversold + price > 12h EMA50 + volume spike
# Short when Williams %R crosses below -20 from overbought + price < 12h EMA50 + volume spike
# Exit when Williams %R returns to neutral zone (-50) or trend reverses
# Designed for low trade frequency (~15-30/year) with mean-reversion edge in ranging markets
# Williams %R identifies exhaustion points, EMA50 filters trend direction, volume confirms conviction

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 6h data for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_6h) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Align Williams %R to main timeframe (6h -> 6h is 1:1, but use align for safety)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        wr = williams_r_aligned[i]
        ema_val = ema_50_aligned[i]
        
        # Williams %R signals: crossover of -80 (oversold) and -20 (overbought)
        wr_prev = williams_r_aligned[i-1]
        wr_cross_above_80 = wr_prev <= -80 and wr > -80
        wr_cross_below_20 = wr_prev >= -20 and wr < -20
        
        # Volume filter: current volume > 1.8 * 20-day average
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80 + uptrend + volume spike
            if wr_cross_above_80 and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R crosses below -20 + downtrend + volume spike
            elif wr_cross_below_20 and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Williams %R returns to neutral (-50) or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Williams %R returns to -50 or trend turns down
                if wr >= -50 or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Williams %R returns to -50 or trend turns up
                if wr <= -50 or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0