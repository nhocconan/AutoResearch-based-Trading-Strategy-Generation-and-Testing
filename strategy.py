#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams %R with 1-day RSI filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions, RSI filters for trend strength,
# volume confirms the strength of the reversal. Works in both bull and bear markets
# by capturing mean reversion within the dominant trend. Target: 20-40 trades/year per symbol.
name = "4h_WilliamsR_RSI1D_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Get 1-day RSI (14-period) for trend filter
    df_1d = get_htf_data(prices, '1d')
    rsi_1d = np.full_like(close, np.nan, dtype=np.float64)
    if len(df_1d) >= 14:
        rsi_14 = pd.Series(df_1d['close']).ewm(span=14, adjust=False, min_periods=14).mean()
        rsi_down = pd.Series(df_1d['close']).ewm(span=14, adjust=False, min_periods=14).mean()
        # Actually compute RSI properly
        delta = pd.Series(df_1d['close']).diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss
        rsi_1d_raw = 100 - (100 / (1 + rs))
        rsi_1d = align_htf_to_ltf(prices, df_1d, rsi_1d_raw.values)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or np.isnan(rsi_1d[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r[i]
        rsi = rsi_1d[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80), RSI > 50 (bullish bias), volume spike
            if (wr < -80 and rsi > 50 and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20), RSI < 50 (bearish bias), volume spike
            elif (wr > -20 and rsi < 50 and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns to overbought (> -20) or RSI < 40
            if wr > -20 or rsi < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns to oversold (< -80) or RSI > 60
            if wr < -80 or rsi > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals