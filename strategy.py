#!/usr/bin/env python3
"""
Hypothesis: Daily price closes outside the weekly Bollinger Bands (20,2) signal strong momentum
continuation in the direction of the breakout. Weekly Bollinger Bands act as dynamic support/
resistance on the daily chart. A close above the upper band indicates bullish momentum; a close
below the lower band indicates bearish momentum. Positions are held until price reverts to the
weekly middle band (20-period SMA), capturing trending moves while avoiding whipsaws in ranging
markets. Volume confirmation ensures breakouts have conviction. Designed for 1d timeframe to
capture multi-day trends with low turnover (~10-20 trades/year) suitable for bear markets like
2025.
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
    
    # Get weekly data for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Bollinger Bands (20,2)
    weekly_close = df_1w['close'].values
    sma_20 = pd.Series(weekly_close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(weekly_close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    middle_band = sma_20  # 20-period SMA
    
    # Align weekly bands to daily timeframe (waits for weekly bar to close)
    upper_bb_daily = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_bb_daily = align_htf_to_ltf(prices, df_1w, lower_band)
    middle_bb_daily = align_htf_to_ltf(prices, df_1w, middle_band)
    
    # Volume confirmation: 20-day volume average
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(upper_bb_daily[i]) or np.isnan(lower_bb_daily[i]) or
            np.isnan(middle_bb_daily[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: close above weekly upper band with volume confirmation
            if price > upper_bb_daily[i] and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: close below weekly lower band with volume confirmation
            elif price < lower_bb_daily[i] and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to weekly middle band
            if price < middle_bb_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to weekly middle band
            if price > middle_bb_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyBB_Breakout_MeanReversion"
timeframe = "1d"
leverage = 1.0