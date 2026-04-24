#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 12h EMA trend filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 12h for EMA trend direction.
- EMA50 > EMA200 on 12h indicates bullish trend (favor longs), EMA50 < EMA200 indicates bearish trend (favor shorts).
- Entry: Long when price breaks above Camarilla H3 level AND 12h EMA50 > EMA200.
         Short when price breaks below Camarilla L3 level AND 12h EMA50 < EMA200.
- Exit: Opposite Camarilla breakout (H3 for shorts, L3 for longs) or EMA trend flip.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
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
    
    # Get 12h data for EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 and EMA200 on 12h
    close_12h = pd.Series(df_12h['close'])
    ema50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_12h = close_12h.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h EMA to 4h
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    # Camarilla levels (based on previous day's OHLC) on 4h
    # Need to group by day to get daily OHLC
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
    daily_ohlc['H3'] = daily_ohlc['close'] + 1.1 * (daily_ohlc['high'] - daily_ohlc['low']) / 6
    daily_ohlc['L3'] = daily_ohlc['close'] - 1.1 * (daily_ohlc['high'] - daily_ohlc['low']) / 6
    daily_ohlc['H4'] = daily_ohlc['close'] + 1.1 * (daily_ohlc['high'] - daily_ohlc['low']) / 2
    daily_ohlc['L4'] = daily_ohlc['close'] - 1.1 * (daily_ohlc['high'] - daily_ohlc['low']) / 2
    
    # Map daily Camarilla levels to 4h bars
    camarilla_map = {}
    for _, row in daily_ohlc.iterrows():
        camarilla_map[row['date']] = {
            'H3': row['H3'],
            'L3': row['L3'],
            'H4': row['H4'],
            'L4': row['L4']
        }
    
    # Vectorized mapping
    dates = prices_df['open_time'].dt.date
    H3 = dates.map(lambda d: camarilla_map.get(d, {}).get('H3', np.nan)).values
    L3 = dates.map(lambda d: camarilla_map.get(d, {}).get('L3', np.nan)).values
    H4 = dates.map(lambda d: camarilla_map.get(d, {}).get('H4', np.nan)).values
    L4 = dates.map(lambda d: camarilla_map.get(d, {}).get('L4', np.nan)).values
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough 12h bars for EMA and 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(ema200_12h_aligned[i]) or 
            np.isnan(H3[i]) or np.isnan(L3[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                ema50 = ema50_12h_aligned[i]
                ema200 = ema200_12h_aligned[i]
                
                # Bullish breakout: price breaks above H3 AND 12h EMA50 > EMA200 (bullish trend)
                if curr_high > H3[i] and ema50 > ema200:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below L3 AND 12h EMA50 < EMA200 (bearish trend)
                elif curr_low < L3[i] and ema50 < ema200:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below L3 OR 12h EMA trend flips to bearish
            if curr_low < L3[i] or ema50_12h_aligned[i] < ema200_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H3 OR 12h EMA trend flips to bullish
            if curr_high > H3[i] or ema50_12h_aligned[i] > ema200_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_12hEMATrend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0