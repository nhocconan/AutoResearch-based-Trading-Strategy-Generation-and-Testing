#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R(14) extreme reversal with 12h EMA(50) trend filter and volume confirmation.
- Primary timeframe: 4h for entries/exits.
- HTF: 12h EMA(50) for trend direction (bullish if price > EMA50, bearish if price < EMA50).
- Williams %R: Long when %R crosses above -80 from below (oversold bounce).
               Short when %R crosses below -20 from above (overbought rejection).
- Volume: Current 4h volume > 1.5 * 20-period volume MA to confirm momentum.
- Entry: Long when Williams %R crosses above -80 AND 12h EMA50 trend bullish AND volume confirmation.
         Short when Williams %R crosses below -20 AND 12h EMA50 trend bearish AND volume confirmation.
- Exit: Opposite Williams %R extreme (%R < -80 for longs, %R > -20 for shorts) or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 60-120 total trades over 4 years (15-30/year) for 4h timeframe.
- Why it works in both bull/bear: Williams %R captures mean-reversion swings within trends,
  while 12h EMA filter ensures we trade with the higher timeframe momentum, reducing whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R(14) on 4h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 12h data for EMA(50) trend and volume MA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h close
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period volume MA on 12h
    vol_ma_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Volume confirmation: current 4h volume > 1.5 * 20-period 12h volume MA (aligned)
    volume_confirm = volume > (1.5 * vol_ma_12h_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14)  # Need enough 12h bars for EMA50 and 14 bars for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_50_val = ema_50_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        wr = williams_r[i]
        vol_ok = volume_confirm[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation
            if vol_ok:
                # Bullish entry: Williams %R crosses above -80 from below (oversold bounce)
                if i > start_idx and williams_r[i-1] <= -80 and wr > -80 and ema_50_val > 0 and curr_close > ema_50_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Williams %R crosses below -20 from above (overbought rejection)
                elif i > start_idx and williams_r[i-1] >= -20 and wr < -20 and ema_50_val > 0 and curr_close < ema_50_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R rises above -80 (overbought) OR loss of volume confirmation
            if wr >= -80 or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R falls below -20 (oversold) OR loss of volume confirmation
            if wr <= -20 or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR14_12hEMA50Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0