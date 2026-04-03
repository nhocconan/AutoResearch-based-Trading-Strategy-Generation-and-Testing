#!/usr/bin/env python3
"""
Experiment #287: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation

HYPOTHESIS: Donchian channel breakouts on 6h timeframe capture medium-term momentum, while weekly pivot points provide institutional reference levels for breakout validation. Volume confirmation filters false breakouts. This strategy targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag while capturing sustained moves in both bull and bear markets. Weekly pivot direction acts as a regime filter: only take long breakouts above weekly pivot, short breakouts below weekly pivot.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from 1d data (using last week's OHLC)
    weekly_pivot = np.full(len(df_1d), np.nan)
    weekly_r1 = np.full(len(df_1d), np.nan)
    weekly_s1 = np.full(len(df_1d), np.nan)
    
    if len(df_1d) >= 5:
        # Group by week (assuming 5 trading days per week)
        for i in range(4, len(df_1d)):
            # Get previous week's OHLC (5 days ago to 1 day ago)
            week_high = np.max(df_1d['high'].iloc[i-5:i])
            week_low = np.min(df_1d['low'].iloc[i-5:i])
            week_close = df_1d['close'].iloc[i-1]
            
            # Calculate pivot points
            pivot = (week_high + week_low + week_close) / 3.0
            r1 = 2 * pivot - week_low
            s1 = 2 * pivot - week_high
            
            weekly_pivot[i] = pivot
            weekly_r1[i] = r1
            weekly_s1[i] = s1
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 50-period EMA on weekly close for trend filter
    weekly_ema50 = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 50:
        close_1w = df_1w['close'].values
        weekly_ema50 = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align weekly EMA to 6h timeframe
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # === 6h Indicators ===
    # Donchian Channel(20)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    if n >= 20:
        # Highest high over past 20 periods
        for i in range(20, n):
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Volume average for confirmation
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(weekly_ema50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Weekly Pivot Direction Filter ---
        # Only trade in direction of weekly pivot relative to weekly EMA
        weekly_bullish = weekly_ema50_aligned[i] > weekly_pivot_aligned[i]
        weekly_bearish = weekly_ema50_aligned[i] < weekly_pivot_aligned[i]
        
        # --- Volume Confirmation ---
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        # --- Donchian Breakout Signals ---
        # Long breakout: price breaks above Donchian high with volume and weekly bullish bias
        long_breakout = (close[i] > donchian_high[i]) and volume_surge and weekly_bullish
        
        # Short breakout: price breaks below Donchian low with volume and weekly bearish bias
        short_breakout = (close[i] < donchian_low[i]) and volume_surge and weekly_bearish
        
        signals[i] = 0.0  # Default to flat
        
        if long_breakout:
            signals[i] = SIZE
        elif short_breakout:
            signals[i] = -SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #287: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation

HYPOTHESIS: Donchian channel breakouts on 6h timeframe capture medium-term momentum, while weekly pivot points provide institutional reference levels for breakout validation. Volume confirmation filters false breakouts. This strategy targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag while capturing sustained moves in both bull and bear markets. Weekly pivot direction acts as a regime filter: only take long breakouts above weekly pivot, short breakouts below weekly pivot.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from 1d data (using last week's OHLC)
    weekly_pivot = np.full(len(df_1d), np.nan)
    weekly_r1 = np.full(len(df_1d), np.nan)
    weekly_s1 = np.full(len(df_1d), np.nan)
    
    if len(df_1d) >= 5:
        # Group by week (assuming 5 trading days per week)
        for i in range(4, len(df_1d)):
            # Get previous week's OHLC (5 days ago to 1 day ago)
            week_high = np.max(df_1d['high'].iloc[i-5:i])
            week_low = np.min(df_1d['low'].iloc[i-5:i])
            week_close = df_1d['close'].iloc[i-1]
            
            # Calculate pivot points
            pivot = (week_high + week_low + week_close) / 3.0
            r1 = 2 * pivot - week_low
            s1 = 2 * pivot - week_high
            
            weekly_pivot[i] = pivot
            weekly_r1[i] = r1
            weekly_s1[i] = s1
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 50-period EMA on weekly close for trend filter
    weekly_ema50 = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 50:
        close_1w = df_1w['close'].values
        weekly_ema50 = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align weekly EMA to 6h timeframe
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # === 6h Indicators ===
    # Donchian Channel(20)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    if n >= 20:
        # Highest high over past 20 periods
        for i in range(20, n):
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Volume average for confirmation
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(weekly_ema50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Weekly Pivot Direction Filter ---
        # Only trade in direction of weekly pivot relative to weekly EMA
        weekly_bullish = weekly_ema50_aligned[i] > weekly_pivot_aligned[i]
        weekly_bearish = weekly_ema50_aligned[i] < weekly_pivot_aligned[i]
        
        # --- Volume Confirmation ---
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        # --- Donchian Breakout Signals ---
        # Long breakout: price breaks above Donchian high with volume and weekly bullish bias
        long_breakout = (close[i] > donchian_high[i]) and volume_surge and weekly_bullish
        
        # Short breakout: price breaks below Donchian low with volume and weekly bearish bias
        short_breakout = (close[i] < donchian_low[i]) and volume_surge and weekly_bearish
        
        signals[i] = 0.0  # Default to flat
        
        if long_breakout:
            signals[i] = SIZE
        elif short_breakout:
            signals[i] = -SIZE
    
    return signals