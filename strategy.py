#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R(14) mean reversion with 12h EMA(34) trend filter and 1d volume spike confirmation.
- Primary timeframe: 6h for Williams %R calculation and entries/exits.
- HTF trend: 12h EMA(34) - bullish when price > EMA34, bearish when price < EMA34.
- Volume confirmation: Current 6h volume > 2.0 * 20-period 1d volume MA to filter low-quality signals.
- Entry logic: 
  * Long when Williams %R crosses above -80 (from oversold) AND 12h EMA34 trend bullish AND volume spike.
  * Short when Williams %R crosses below -20 (from overbought) AND 12h EMA34 trend bearish AND volume spike.
- Exit: Opposite Williams %R cross (%R crosses below -50 for longs, above -50 for shorts) or loss of volume/Trend.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Why should work: Williams %R excels at catching reversals in ranging markets (common in 2025 BTC/ETH),
  while 12h EMA34 trend filter avoids counter-trend trades during strong moves. Volume spike confirms
  participation, reducing false signals. Mean reversion on 6h timeframe balances trade frequency and accuracy.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R(14) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 12h data for EMA(34) trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 12h close
    ema_34 = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 1d data for volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period volume MA on 1d
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 6h volume > 2.0 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # Need enough bars for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_34_val = ema_34_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        wr = williams_r[i]
        wr_prev = williams_r[i-1]
        
        if position == 0:
            # Check for entry signals with volume spike and trend alignment
            if volume_spike[i]:
                # Bullish: Williams %R crosses above -80 (from oversold) AND 12h EMA34 bullish (price > EMA34)
                if wr > -80 and wr_prev <= -80 and curr_close > ema_34_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Williams %R crosses below -20 (from overbought) AND 12h EMA34 bearish (price < EMA34)
                elif wr < -20 and wr_prev >= -20 and curr_close < ema_34_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 OR loss of volume confirmation OR loss of trend
            if wr < -50 and wr_prev >= -50 or not volume_spike[i] or curr_close < ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 OR loss of volume confirmation OR loss of trend
            if wr > -50 and wr_prev <= -50 or not volume_spike[i] or curr_close > ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR14_12hEMA34Trend_1dVolumeSpike_v1"
timeframe = "6h"
leverage = 1.0