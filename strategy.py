#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d trend filter and volume confirmation
# Long when: Williams %R < -80 (oversold), price > 1d EMA(50), and volume > 1.5x 20-period average
# Short when: Williams %R > -20 (overbought), price < 1d EMA(50), and volume > 1.5x 20-period average
# Exit when Williams %R crosses back to -50
# Williams %R identifies reversals, EMA(50) filters trend direction, volume confirms strength
# Designed for ~15-25 trades/year per symbol (60-100 total over 4 years)

name = "12h_WilliamsR_EMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_1d_50_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_1d = ema_1d_50_aligned[i]
        vol_ma = vol_ma_20[i]
        vol_ratio = volume[i] / (vol_ma + 1e-10)
        wr = williams_r[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold), price above 1d EMA50, high volume
            if wr < -80 and price > ema_1d and vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought), price below 1d EMA50, high volume
            elif wr > -20 and price < ema_1d and vol_ratio > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R > -50 (returning to neutral)
            if wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R < -50 (returning to neutral)
            if wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals