#!/usr/bin/env python3
"""
Experiment #5276: 12h Donchian(20) breakout + 1d EMA50 trend + volume spike filter
HYPOTHESIS: On 12h timeframe, Donchian breakouts capture strong momentum moves. 
Filtering by 1d EMA50 ensures we only trade in the direction of the daily trend, 
avoiding counter-trend whipsaws. Volume spike confirmation adds conviction to breakouts. 
This combination should work in both bull and bear markets by following the dominant 
daily trend while avoiding false breakouts in ranging conditions. 
Designed for 12-37 trades/year on 12h timeframe (50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5276_12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
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
    
    # === HTF: 1d data for EMA50 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        ema_50 = pd.Series(df_1d['close']).ewm(span=50, min_periods=50, adjust=False).mean().shift(1).values
        ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    else:
        ema_50_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Donchian Channel (20-period) ===
    # Upper band: 20-period high
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Middle band: 20-period average (optional for exit)
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # === 12h Indicators: Volume Spike (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma  # Current volume relative to average
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 50)  # Donchian, EMA50 warmup
    
    for i in range(warmup, n):
        # --- Session Filter: 00-24 UTC (12h timeframe, full day) ---
        # 12h candles already cover major sessions, no intraday filter needed
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position when price crosses Donchian middle or trend reverses ---
        if in_position:
            # Check trend consistency
            trend_bullish = price > ema_50_aligned[i]
            trend_bearish = price < ema_50_aligned[i]
            
            # Exit conditions:
            # 1. Price crosses Donchian middle (mean reversion)
            # 2. Daily trend reverses
            if position_side > 0:  # Long position
                if (price < donchian_mid[i]) or (not trend_bullish):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                if (price > donchian_mid[i]) or (not trend_bearish):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Donchian breakout
        breakout_up = price > donchian_high[i]   # Price breaks above upper band
        breakout_down = price < donchian_low[i]  # Price breaks below lower band
        
        # Trend filter from 1d EMA50
        trend_bullish = price > ema_50_aligned[i]
        trend_bearish = price < ema_50_aligned[i]
        
        # Volume confirmation: spike above 1.5x average volume
        volume_spike = vol_ratio[i] > 1.5
        
        # Entry conditions: Breakout + trend alignment + volume confirmation
        if breakout_up and trend_bullish and volume_spike:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif breakout_down and trend_bearish and volume_spike:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals