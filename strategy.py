#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h EMA trend filter and volume confirmation.
# Long when: Williams %R crosses above -80 from below (oversold reversal) AND price > 12h EMA50 AND volume > 1.5x 20-period average
# Short when: Williams %R crosses below -20 from above (overbought reversal) AND price < 12h EMA50 AND volume > 1.5x 20-period average
# Exit when: Williams %R returns to -50 (mean reversion) OR opposite extreme is touched
# Williams %R identifies short-term reversals, 12h EMA50 filters for trend direction, volume confirms momentum.
# Designed for ~15-25 trades/year per symbol with controlled risk in both bull and bear markets.
name = "6h_WilliamsR_EMA50_VolumeFilter"
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
    
    # 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R (14-period) on 6x data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume average (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        wr = williams_r[i]
        ema_50 = ema_50_12h_aligned[i]
        
        # Previous Williams %R for crossover detection
        wr_prev = williams_r[i-1]
        
        if position == 0:
            # Long entry: Williams %R crosses above -80 from below (bullish reversal) + uptrend + volume
            if wr_prev <= -80 and wr > -80 and price > ema_50 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R crosses below -20 from above (bearish reversal) + downtrend + volume
            elif wr_prev >= -20 and wr < -20 and price < ema_50 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns to -50 (mean reversion) OR touches -20 (overbought)
            if wr >= -50 or wr >= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns to -50 (mean reversion) OR touches -80 (oversold)
            if wr <= -50 or wr <= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals