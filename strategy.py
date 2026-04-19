#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1-week RSI filter and 6-hour Williams %R extremes.
# Uses weekly RSI(14) to filter trend direction (bull when >50, bear when <50) and
# enters on 6-hour Williams %R extremes (<20 for long, >80 for short) with volume confirmation.
# Designed to capture mean reversion within the weekly trend, reducing false signals in chop.
# Target: 15-25 trades/year to avoid fee drag.
name = "6h_1w_RSI50_WilliamsR_Volume"
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
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for RSI(14) filter (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Calculate RSI(14)
    delta = pd.Series(close_1w).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1w = 100 - (100 / (1 + rs))
    rsi_14_1w = rsi_14_1w.fillna(50).values  # Neutral when insufficient data
    rsi_14_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_14_1w)
    
    # Williams %R(14) on 6h data (called ONCE before loop)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close_6h) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(-50).values  # Neutral when range=0
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(rsi_14_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: weekly bullish (RSI>50) AND Williams %R oversold (<-80) with volume
            if (rsi_14_1w_aligned[i] > 50 and 
                williams_r[i] < -80 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: weekly bearish (RSI<50) AND Williams %R overbought (>-20) with volume
            elif (rsi_14_1w_aligned[i] < 50 and 
                  williams_r[i] > -20 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if weekly turns bearish OR Williams %R exits oversold (>-50)
            if rsi_14_1w_aligned[i] < 50 or williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if weekly turns bullish OR Williams %R exits overbought (<-50)
            if rsi_14_1w_aligned[i] > 50 or williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals