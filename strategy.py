#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1d trend filter and volume confirmation
# Long when Williams %R(14) crosses above -80 (oversold reversal) AND price > 1d EMA50 AND volume > 1.5x 20-bar avg
# Short when Williams %R(14) crosses below -20 (overbought reversal) AND price < 1d EMA50 AND volume > 1.5x 20-bar avg
# Exit when Williams %R returns to -50 (mean reversion center)
# Uses discrete position sizing (0.25) to reduce fee drag and improve test generalization.
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to avoid overtrading.
# Williams %R is a momentum oscillator that identifies overbought/oversold conditions.
# In bear markets (2022, 2025+), it captures sharp bounces from extreme oversold levels during capitulation.
# In bull markets, it catches pullbacks in uptrends before continuation.
# The 1d EMA50 filter ensures we only trade in the direction of the higher timeframe trend.
# Volume confirmation ensures breakouts have participation, reducing false signals.

name = "6h_WilliamsR_Extreme_Reversal_1dEMA50_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R on 6h data: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Williams %R ranges from -100 (oversold) to 0 (overbought)
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: >1.5x 20-bar average volume (balanced to avoid overtrading)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20, 50)  # Williams %R, volume MA, and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_williams_r = williams_r[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Williams %R returns to -50 (mean reversion)
            if curr_williams_r >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R returns to -50 (mean reversion)
            if curr_williams_r <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Williams %R crosses above -80 (oversold reversal) AND price > 1d EMA50 AND volume confirmation
            if curr_williams_r > -80 and curr_williams_r < -50 and curr_close > curr_ema50_1d and vol_conf:
                # Additional check: ensure we're coming from below -80 (crossing up)
                if i > start_idx and williams_r[i-1] <= -80:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            # Short when Williams %R crosses below -20 (overbought reversal) AND price < 1d EMA50 AND volume confirmation
            elif curr_williams_r < -20 and curr_williams_r > -50 and curr_close < curr_ema50_1d and vol_conf:
                # Additional check: ensure we're coming from above -20 (crossing down)
                if i > start_idx and williams_r[i-1] >= -20:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals