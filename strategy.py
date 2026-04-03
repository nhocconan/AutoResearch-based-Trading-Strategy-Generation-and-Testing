#!/usr/bin/env python3
"""
Experiment #282: 12h Donchian(20) breakout + 1d EMA(50) trend + 1w volume spike filter

HYPOTHESIS: Trading Donchian channel breakouts on 12h timeframe with 1d EMA trend alignment and 1w volume confirmation captures medium-term trends while avoiding false breakouts. The 12h timeframe reduces trade frequency to minimize fee drag, EMA(50) ensures we trade with the intermediate trend, and 1w volume spike (>2x average) confirms institutional participation. This combination should work in both bull and bear markets by only taking breakouts in the direction of the 1d trend with volume confirmation. Targets 12-37 trades/year (50-150 total over 4 years) to stay within the proven range for 12h strategies.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_ema_volspike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for volume spike filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate volume average and spike threshold on 1w data
    if len(df_1w) >= 20:
        volume_1w = df_1w['volume'].values
        # 20-period average volume on weekly timeframe
        vol_avg_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
        # Volume spike = current volume > 2x 20-period average
        vol_spike_1w = volume_1w > (2.0 * vol_avg_20)
        # Handle NaN values
        vol_spike_1w = np.where(np.isnan(vol_spike_1w), False, vol_spike_1w)
        vol_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_spike_1w.astype(np.float64))
    else:
        vol_spike_1w_aligned = np.full(n, 0.0)
    
    # === 12h Indicators ===
    # Donchian Channel (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss or mean reversion) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                # Stoploss hit
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit: reverse signal or extended move
                if close[i] < ema_50_1d_aligned[i]:  # Price closes below 1d EMA
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                # Stoploss hit
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit: reverse signal or extended move
                if close[i] > ema_50_1d_aligned[i]:  # Price closes above 1d EMA
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Breakout conditions
        breakout_up = high[i] > highest_high_20[i-1]  # New 20-period high
        breakout_down = low[i] < lowest_low_20[i-1]   # New 20-period low
        
        # Trend filter: price relative to 1d EMA
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: 1w volume spike
        vol_spike = vol_spike_1w_aligned[i] > 0.5  # True if aligned array shows spike
        
        # Long: Bullish breakout with volume confirmation and price above 1d EMA
        if breakout_up and price_above_ema and vol_spike:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        # Short: Bearish breakout with volume confirmation and price below 1d EMA
        elif breakout_down and price_below_ema and vol_spike:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals