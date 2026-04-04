#!/usr/bin/env python3
"""
exp_6451_6h_donchian20_1d_weekly_pivot_vol_v2
Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction filter and volume confirmation.
- Primary TF: 6h for entry timing
- HTF: 1d for weekly pivot levels (R1,R2,S1,S2) and trend filter
- Weekly pivot: calculated from prior week's (Sunday UTC 00:00) high/low/close
- Long: price breaks above Donchian(20) high AND price > weekly pivot point AND volume > 1.5x 20-period average
- Short: price breaks below Donchian(20) low AND price < weekly pivot point AND volume > 1.5x 20-period average
- Weekly pivot acts as regime filter: above = bullish bias (longs only), below = bearish bias (shorts only)
- Volume confirmation reduces false breakouts
- Position size: 0.25 discrete levels to minimize fee churn
- Stoploss: signal -> 0 when price moves 2*ATR(22) against position
- Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6451_6h_donchian20_1d_weekly_pivot_vol_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from prior week (Sunday UTC 00:00)
    # We need to group 1d data by week starting Sunday
    df_1d = df_1d.copy()
    df_1d['week_start'] = df_1d.index.to_series().dt.to_period('W-SUN').dt.start_time
    weekly = df_1d.groupby('week_start').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    # Calculate weekly pivot: P = (H + L + C) / 3
    weekly['pivot'] = (weekly['high'] + weekly['low'] + weekly['close']) / 3.0
    
    # Forward fill weekly pivot to each 1d bar (each day gets its week's pivot)
    weekly_pivot_series = pd.Series(index=df_1d.index, dtype=float)
    for _, row in weekly.iterrows():
        week_start = row['week_start']
        # Find all 1d bars in this week
        mask = (df_1d.index >= week_start) & (df_1d.index < week_start + pd.Timedelta(days=7))
        weekly_pivot_series.loc[mask] = row['pivot']
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_values = weekly_pivot_series.values
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot_values)
    
    # Calculate Donchian channels on 6h
    lookback = 20
    high_roll = prices['high'].rolling(window=lookback, min_periods=lookback).max()
    low_roll = prices['low'].rolling(window=lookback, min_periods=lookback).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    
    # Calculate ATR for stoploss
    atr_period = 22
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift())
    low_close = np.abs(prices['low'] - prices['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean()
    vol_ratio = prices['volume'] / vol_ma
    vol_ratio = vol_ratio.fillna(0.0).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(lookback, n):
        # Skip if weekly pivot not available (first week)
        if np.isnan(weekly_pivot_aligned[i]):
            continue
            
        # Get current values
        price = prices['close'].iloc[i]
        vol_ok = vol_ratio[i] > 1.5
        
        # Long condition: break above Donchian high AND price above weekly pivot AND volume OK
        if price > donchian_high[i] and price > weekly_pivot_aligned[i] and vol_ok:
            if position <= 0:  # Flip from short/flat to long
                signals[i] = 0.25
                position = 1
                entry_price = price
            else:
                signals[i] = 0.0  # Already long
        # Short condition: break below Donchian low AND price below weekly pivot AND volume OK
        elif price < donchian_low[i] and price < weekly_pivot_aligned[i] and vol_ok:
            if position >= 0:  # Flip from long/flat to short
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0  # Already short
        else:
            # Check stoploss
            if position == 1:  # Long position
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0  # Stoploss hit
                    position = 0
                else:
                    signals[i] = 0.25  # Keep long
            elif position == -1:  # Short position
                if price > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0  # Stoploss hit
                    position = 0
                else:
                    signals[i] = -0.25  # Keep short
            else:
                signals[i] = 0.0  # Flat
    
    # Ensure no look-ahead: at bar i, we only used data up to i
    # Donchian uses min_periods=lookback so first lookback-1 values are NaN
    # Weekly pivot aligned uses shift(1) internally in align_htf_to_ltf
    # Volume ratio uses min_periods=20
    # ATR uses min_periods=atr_period in ewm
    
    return signals