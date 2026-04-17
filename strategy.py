#!/usr/bin/env python3
"""
Hypothesis: Daily 1d strategy using weekly Donchian breakout with volume confirmation and 1d EMA50 trend filter.
- Enter long when price breaks above weekly Donchian high (20) + volume > 1.5x 20-day volume MA + price above daily EMA50
- Enter short when price breaks below weekly Donchian low (20) + volume > 1.5x 20-day volume MA + price below daily EMA50
- Exit when price crosses back inside weekly Donchian channel
- Fixed position size 0.25 to manage drawdown
- Designed for 1d timeframe with strict entry conditions to limit trades to 30-100 total over 4 years
- Weekly Donchian captures long-term structure, effective in both trending and ranging markets
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
    
    # Daily EMA50 trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean()
    
    # Daily volume MA(20)
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Weekly Donchian (20) - using weekly high/low
    df_1w = get_htf_data(prices, '1w')
    donch_high_20 = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max()
    donch_low_20 = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min()
    
    # Align weekly Donchian to daily timeframe (wait for weekly bar close)
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high_20.values)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low_20.values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50.iloc[i]) or np.isnan(volume_ma_20.iloc[i]) or 
            np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        upper = donch_high_aligned[i]
        lower = donch_low_aligned[i]
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
            # Exit when price crosses back inside weekly Donchian channel (mean reversion)
            if price < upper and price > lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses back inside weekly Donchian channel (mean reversion)
            if price < upper and price > lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchianBreakout_VolumeTrend"
timeframe = "1d"
leverage = 1.0