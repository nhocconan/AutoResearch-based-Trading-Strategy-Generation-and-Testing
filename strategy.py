#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R + 1d EMA34 trend + volume spike.
- Primary timeframe: 6h for entries/exits.
- HTF: 1d EMA34 for trend direction (bullish if price > EMA34, bearish if price < EMA34).
- Volume: Current 6h volume > 2.0 * 20-period volume MA to filter low-quality breakouts.
- Entry: Long when Williams %R(14) crosses above -80 (oversold bounce) AND 1d EMA34 bullish (price > EMA) AND volume spike.
         Short when Williams %R(14) crosses below -20 (overbought rejection) AND 1d EMA34 bearish (price < EMA) AND volume spike.
- Exit: Opposite Williams %R level (long exits at -20, short exits at -80) or loss of volume confirmation or trend reversal.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Williams %R captures short-term momentum extremes. Combined with 1d trend filter and volume confirmation, 
this avoids counter-trend trades and works in both bull and bear markets by only taking trades in the 
direction of the 1d trend. The volume spike requirement ensures participation, reducing false signals.
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
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    # Avoid division by zero
    hh_ll = highest_high - lowest_low
    williams_r = np.where(hh_ll != 0, (highest_high - close) / hh_ll * -100, -50)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 1d
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 6h volume > 2.0 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # Need enough bars for EMA34, volume MA, and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(williams_r[i]) or i == 0):  # i==0 to avoid lookback issues
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        ema_val = ema_1d_aligned[i]
        curr_wr = williams_r[i]
        prev_wr = williams_r[i-1] if i > 0 else -50
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: Williams %R crosses above -80 (oversold bounce) AND 1d EMA34 bullish (price > EMA)
                if prev_wr <= -80 and curr_wr > -80 and curr_close > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Williams %R crosses below -20 (overbought rejection) AND 1d EMA34 bearish (price < EMA)
                elif prev_wr >= -20 and curr_wr < -20 and curr_close < ema_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -20 (overbought) OR loss of volume confirmation OR trend reversal
            if curr_wr >= -20 or not volume_spike[i] or curr_close < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -80 (oversold) OR loss of volume confirmation OR trend reversal
            if curr_wr <= -80 or not volume_spike[i] or curr_close > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0