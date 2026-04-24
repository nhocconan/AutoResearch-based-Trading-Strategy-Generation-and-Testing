#!/usr/bin/env python3
"""
Hypothesis: 1h EMA crossover with 4h Donchian trend filter and volume spike confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h Donchian(20) for trend direction (bullish if close > upper, bearish if close < lower).
- EMA crossover: 9/21 EMA on 1h for momentum entry timing.
- Volume confirmation: current volume > 1.5 * 20-period volume MA.
- Entry: Long when 9 EMA > 21 EMA AND 4h trend bullish AND volume confirmed.
         Short when 9 EMA < 21 EMA AND 4h trend bearish AND volume confirmed.
- Exit: Opposite EMA crossover.
- Signal size: 0.20 discrete to minimize fee churn and control drawdown.
Designed to work in both bull and bear markets via 4h trend filter and volatility-adjusted entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Upper band: highest high over 20 periods
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over 20 periods
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to 1h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    
    # Calculate 1h EMAs for entry timing
    close_series = pd.Series(close)
    ema_9 = close_series.ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_21 = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 20, 21, 20)  # Need enough bars for Donchian, EMAs, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema_9[i]) or np.isnan(ema_21[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            
            # Determine 4h trend: bullish if close > upper band, bearish if close < lower band
            # Note: Using 4h close aligned to 1h for trend determination
            htf_close_aligned = align_htf_to_ltf(prices, df_4h, close_4h)
            htf_close = htf_close_aligned[i]
            
            trend_bullish = htf_close > upper_20_aligned[i]
            trend_bearish = htf_close < lower_20_aligned[i]
            
            # Long: 9 EMA > 21 EMA AND 4h trend bullish AND volume confirmed
            if ema_9[i] > ema_21[i] and trend_bullish and vol_confirmed:
                signals[i] = 0.20
                position = 1
            # Short: 9 EMA < 21 EMA AND 4h trend bearish AND volume confirmed
            elif ema_9[i] < ema_21[i] and trend_bearish and vol_confirmed:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long when 9 EMA < 21 EMA (momentum shift)
            if ema_9[i] < ema_21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short when 9 EMA > 21 EMA (momentum shift)
            if ema_9[i] > ema_21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA9_21_4hDonchian20_Trend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0