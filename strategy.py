#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using weekly Donchian breakout with volume confirmation and trend filter.
- Enter long when price breaks above weekly Donchian high + volume > 1.5x 20-period volume MA + price above daily EMA50
- Enter short when price breaks below weekly Donchian low + volume > 1.5x 20-period volume MA + price below daily EMA50
- Exit when price crosses back inside weekly Donchian channel
- Fixed position size 0.25 to manage drawdown
- Designed for 1d timeframe with strict entry conditions to limit trades to 30-100 total over 4 years
- Uses weekly structure for direction and daily timeframe for entry timing and volume confirmation
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
    
    # Weekly Donchian channel (20 periods)
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    donchian_high = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Trend filter: daily EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_ma_20.iloc[i]) or np.isnan(ema_50.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        ema_val = ema_50.iloc[i]
        
        if position == 0:
            # Look for weekly Donchian breakouts with volume confirmation and trend filter
            # Long: price breaks above weekly Donchian high + volume spike + price above EMA50
            if price > upper and vol > 1.5 * vol_ma and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low + volume spike + price below EMA50
            elif price < lower and vol > 1.5 * vol_ma and price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price crosses back inside weekly Donchian channel
            if price < upper and price > lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses back inside weekly Donchian channel
            if price < upper and price > lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchianBreakout_VolumeTrend"
timeframe = "1d"
leverage = 1.0