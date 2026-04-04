#!/usr/bin/env python3
"""
Experiment #5279: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot direction (price > weekly pivot = bullish bias, price < weekly pivot = bearish bias) and confirmed by volume spikes (>1.5x average volume) capture strong directional moves while minimizing false breakouts. Weekly pivot provides structural support/resistance from higher timeframe, reducing whipsaws in ranging markets. Designed for 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to balance opportunity with fee drag control. Works in bull markets by buying breakouts above weekly pivot and in bear markets by selling breakdowns below weekly pivot, avoiding trades when price is near weekly pivot (range) where breakouts fail.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5279_6h_donchian_breakout_weekly_pivot_vol_v1"
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
    
    # === HTF: 1w data for weekly pivot (using 1d as proxy since 1w not in standard TFs) ===
    # Using 1d to calculate weekly pivot points (standard practice: weekly pivot from prior week's daily OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 1:
        # Calculate weekly pivot from prior week's daily data (using last 5 trading days approximate)
        # Weekly Pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
        # We'll use rolling window of 5 days to approximate weekly OHLC
        if len(df_1d) >= 5:
            weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1).values  # Prior week high
            weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1).values      # Prior week low  
            weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().shift(1).values # Prior week close
            weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
            weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
        else:
            weekly_pivot_aligned = np.full(n, np.nan)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    # Upper band: 20-period high
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    # Lower band: 20-period low  
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 6h Indicators: Volume Confirmation ===
    # Volume spike: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 20, 20, 5)  # Donchian, volume avg, weekly pivot warmup
    
    for i in range(warmup, n):
        # --- Session Filter: 00-24 UTC (6h timeframe, use full day) ---
        # 6h candles already filter to specific sessions, so we can use full day
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # --- Exit Logic: Close position when price retouches Donchian opposite side or weekly pivot ---
        if in_position:
            # Exit conditions:
            # 1. Price retouches opposite Donchian band (mean reversion signal)
            # 2. Price crosses weekly pivot (regime change)
            # 3. Volume drops significantly (loss of momentum)
            if position_side > 0:  # Long position
                if (price <= donchian_lower[i]) or (price < weekly_pivot_aligned[i]) or (volume < 0.5 * avg_volume[i]):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                if (price >= donchian_upper[i]) or (price > weekly_pivot_aligned[i]) or (volume < 0.5 * avg_volume[i]):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Donchian breakout conditions
        breakout_up = price > donchian_upper[i]   # Bullish breakout above upper band
        breakout_down = price < donchian_lower[i] # Bearish breakdown below lower band
        
        # Weekly pivot direction filter
        above_weekly_pivot = price > weekly_pivot_aligned[i]  # Bullish bias
        below_weekly_pivot = price < weekly_pivot_aligned[i]  # Bearish bias
        
        # Volume confirmation
        vol_confirmed = volume_spike[i]  # Volume spike confirms breakout strength
        
        # Entry conditions: Donchian breakout + weekly pivot alignment + volume confirmation
        if breakout_up and above_weekly_pivot and vol_confirmed:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif breakout_down and below_weekly_pivot and vol_confirmed:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals