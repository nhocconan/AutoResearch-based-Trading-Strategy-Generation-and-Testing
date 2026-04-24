#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla Pivot Breakout with 4h Trend Filter and Volume Spike.
- Camarilla pivot levels (H3, L3, H4, L4) identify intraday support/resistance.
- Breakout of H3/L3 with volume confirmation captures momentum moves.
- 4h EMA50 provides higher-timeframe trend filter to align with dominant trend.
- Session filter (08-20 UTC) reduces noise during low-liquidity hours.
- Position size 0.20 balances profit and drawdown control.
- Target trades: 80-160 total over 4 years (20-40/year) to balance opportunity and fee drag.
- Works in bull/bear markets via 4h trend filter and volatility expansion logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivots using previous day's OHLC
    # We need to group by day to get daily OHLC
    prices_df = prices.copy()
    prices_df['date'] = prices_df['open_time'].dt.date
    daily_ohlc = prices_df.groupby('date').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    # Map daily OHLC to each 1h bar
    daily_high_map = dict(zip(daily_ohlc['date'], daily_ohlc['high']))
    daily_low_map = dict(zip(daily_ohlc['date'], daily_ohlc['low']))
    daily_close_map = dict(zip(daily_ohlc['date'], daily_ohlc['close']))
    
    prev_high = prices_df['date'].map(daily_high_map.shift(1))
    prev_low = prices_df['date'].map(daily_low_map.shift(1))
    prev_close = prices_df['date'].map(daily_close_map.shift(1))
    
    # Calculate Camarilla levels
    H3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    L3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    H4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    L4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Volume confirmation: > 2.0x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    # Start from index where all indicators are ready
    start_idx = 24  # Need at least 24 bars for volume MA and previous day data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(prev_high.iloc[i]) or np.isnan(prev_low.iloc[i]) or 
            np.isnan(prev_close.iloc[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get Camarilla levels for current bar
        h3 = H3.iloc[i]
        l3 = L3.iloc[i]
        h4 = H4.iloc[i]
        l4 = L4.iloc[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Only trade with volume confirmation and in session
            if volume_confirm and in_session:
                # Long: break above H3 + above 4h EMA50 (bullish higher-timeframe trend)
                if close[i] > h3 and close[i] > ema_50_4h_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                # Short: break below L3 + below 4h EMA50 (bearish higher-timeframe trend)
                elif close[i] < l3 and close[i] < ema_50_4h_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price closes below L3 OR hits H4 (profit target)
            if close[i] < l3 or close[i] > h4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price closes above H3 OR hits L4 (profit target)
            if close[i] > h3 or close[i] < l4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0