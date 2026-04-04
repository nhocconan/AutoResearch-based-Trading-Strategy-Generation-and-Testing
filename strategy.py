#!/usr/bin/env python3
"""
Experiment #5271: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts capture momentum moves. Filtered by 1d weekly pivot direction (price above/below weekly pivot) ensures we trade with the higher timeframe trend. Volume confirmation (volume > 1.5x 20-period average) adds conviction. In bull regime (price > 1d weekly pivot), we go long on Donchian breakout above upper band with volume confirmation. In bear regime (price < 1d weekly pivot), we go short on Donchian breakdown below lower band with volume confirmation. Uses discrete position sizing (0.25) to balance profit potential with drawdown control. Designed for 12-25 trades/year on 6h timeframe (50-100 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5271_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 1d data for weekly pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 5:
        # Weekly pivot from prior week (using last 5 trading days approx)
        # Use last available complete week's OHLC
        if len(df_1d) >= 5:
            # Get last 5 days (prior week) - assuming we have at least 5 days
            week_high = df_1d['high'].iloc[-5:].max()
            week_low = df_1d['low'].iloc[-5:].min()
            week_close = df_1d['close'].iloc[-5:].iloc[-1]  # Close of 5th day back
            # Weekly pivot = (H + L + C) / 3
            weekly_pivot = (week_high + week_low + week_close) / 3.0
            # For simplicity, use the same pivot value for all bars (updated weekly)
            # In practice, we'd update weekly, but for backtest we use expanding window
            weekly_pivot_series = pd.Series(index=df_1d.index, dtype=float)
            # Calculate weekly pivot for each week
            for j in range(4, len(df_1d)):  # Start from 5th day (index 4)
                week_h = df_1d['high'].iloc[j-4:j+1].max()
                week_l = df_1d['low'].iloc[j-4:j+1].min()
                week_c = df_1d['close'].iloc[j]
                weekly_pivot_series.iloc[j] = (week_h + week_l + week_c) / 3.0
            # Forward fill for incomplete weeks
            weekly_pivot_series = weekly_pivot_series.ffill()
            weekly_pivot_values = weekly_pivot_series.values
            weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot_values)
        else:
            weekly_pivot_aligned = np.full(n, np.nan)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20) ===
    # Upper band: 20-period high
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)  # Avoid division by zero
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 20, 5)  # Donchian, volume MA warmup
    
    for i in range(warmup, n):
        # --- Session Filter: 00-24 UTC (6h timeframe, full day) ---
        # 6h candles already cover major sessions
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = vol_ratio[i] > 1.5  # Volume > 1.5x 20-period average
        
        # --- Exit Logic: Close position when price reverses back into Donchian channel ---
        if in_position:
            # Exit when price crosses back below upper band (for long) or above lower band (for short)
            if position_side > 0:  # Long position
                if price < donchian_upper[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                if price > donchian_lower[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Regime filter from 1d weekly pivot
        regime_bullish = price > weekly_pivot_aligned[i]
        regime_bearish = price < weekly_pivot_aligned[i]
        
        # Entry conditions: Donchian breakout + regime match + volume confirmation
        if regime_bullish and price > donchian_upper[i] and vol_ok:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif regime_bearish and price < donchian_lower[i] and vol_ok:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals