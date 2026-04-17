#!/usr/bin/env python3
"""
12h Williams %R + 1d EMA200 filter + volume confirmation
- Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100 over 14 periods
- Long when Williams %R < -80 (oversold) and price > 1d EMA200 and volume > 1.5x 20-period volume MA
- Short when Williams %R > -20 (overbought) and price < 1d EMA200 and volume > 1.5x 20-period volume MA
- Exit when Williams %R crosses above -50 (for long) or below -50 (for short)
- Position size 0.25 to manage drawdown
- Designed for 12h timeframe with strict entry conditions to limit trades to 50-150 total over 4 years
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1-day EMA200 for trend filter
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Williams %R (14-period)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, period)  # warmup for Williams %R and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_200_aligned[i]) or
            np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        wr = williams_r[i]
        ema_val = ema_200_aligned[i]
        
        if position == 0:
            # Look for Williams %R signals with volume confirmation and trend filter
            # Long: Williams %R < -80 (oversold), price above EMA200, volume spike
            if wr < -80 and price > ema_val and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought), price below EMA200, volume spike
            elif wr > -20 and price < ema_val and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when Williams %R crosses above -50 (moving out of oversold)
            if wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when Williams %R crosses below -50 (moving out of overbought)
            if wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_Volume_1dEMA200"
timeframe = "12h"
leverage = 1.0