#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h for execution, HTF: 12h for EMA trend direction.
- EMA34 > 0 indicates bullish bias, EMA34 < 0 indicates bearish bias.
- Entry: Long when price breaks above Camarilla R3 AND 12h EMA34 > 0 (bullish breakout in uptrend).
         Short when price breaks below Camarilla S3 AND 12h EMA34 < 0 (bearish breakout in downtrend).
- Exit: Opposite Camarilla breakout (R3/S3) or EMA trend flip.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in both bull and bear markets by following the 12h EMA trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA34
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 35:
        return np.zeros(n)
    
    # Calculate EMA34 on 12h
    ema34 = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_12h, ema34)
    
    # Calculate Camarilla levels (R3, S3) from previous day's OHLC on 4h
    # Need to group 4h bars by day to get daily OHLC
    prices_df = prices.copy()
    prices_df['date'] = prices_df['open_time'].dt.date
    daily_ohlc = prices_df.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(daily_ohlc) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    camarilla_r3 = np.full(len(prices), np.nan)
    camarilla_s3 = np.full(len(prices), np.nan)
    
    for i in range(len(daily_ohlc) - 1):
        prev_day = daily_ohlc.iloc[i]
        high_prev = prev_day['high']
        low_prev = prev_day['low']
        close_prev = prev_day['close']
        
        range_prev = high_prev - low_prev
        if range_prev <= 0:
            continue
            
        camarilla_r3_val = close_prev + range_prev * 1.1 / 4
        camarilla_s3_val = close_prev - range_prev * 1.1 / 4
        
        # Apply to next day's 4h bars
        start_idx = prices_df.index[prices_df['date'] == daily_ohlc.iloc[i+1]['date']][0]
        end_idx = start_idx + 24  # 24 * 4h = 96h = 4 days, but we want just next day
        # Actually, we need to find all 4h bars belonging to the next day
        next_day_mask = prices_df['date'] == daily_ohlc.iloc[i+1]['date']
        next_day_indices = prices_df.index[next_day_mask].tolist()
        
        for idx in next_day_indices:
            if idx < len(prices):
                camarilla_r3[idx] = camarilla_r3_val
                camarilla_s3[idx] = camarilla_s3_val
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(35, 20)  # Need enough 12h bars for EMA and 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_trend = ema34_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if ema_trend > 0:  # Bullish trend bias
                    # Bullish breakout: price closes above Camarilla R3
                    if curr_close > camarilla_r3[i]:
                        signals[i] = 0.25
                        position = 1
                elif ema_trend < 0:  # Bearish trend bias
                    # Bearish breakout: price closes below Camarilla S3
                    if curr_close < camarilla_s3[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price closes below Camarilla S3 OR EMA trend flips bearish
            if curr_close < camarilla_s3[i] or ema_trend < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Camarilla R3 OR EMA trend flips bullish
            if curr_close > camarilla_r3[i] or ema_trend > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_12hEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0