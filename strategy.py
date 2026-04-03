#!/usr/bin/env python3
"""
Experiment #232: 12h Donchian Breakout + Volume + Chop Filter

HYPOTHESIS: Donchian(20) breakouts on 12h timeframe with volume confirmation and 
choppiness regime filter captures strong trending moves while avoiding whipsaws in ranging markets. 
In trending regimes (CHOP < 38.2), we trade breakouts in direction of trend. 
In ranging regimes (CHOP > 61.8), we avoid breakout trades to prevent false signals. 
This structure-based approach works in both bull (strong breakouts) and bear (failed reversals) markets. 
12h timeframe targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_232_12h_donchian_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for choppiness regime detection (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Choppiness Index (CHOP)
    def calculate_chop(high, low, close, period=14):
        """Calculate Choppiness Index"""
        tr = np.zeros(len(high))
        for i in range(len(high)):
            if i == 0:
                tr[i] = high[i] - low[i]
            else:
                tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        
        chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        return chop
    
    chop_1d = calculate_chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 12h Indicators: Donchian Channels (20-period) ===
    def calculate_donchian(high, low, period=20):
        """Calculate Donchian Channels"""
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_20_upper, donch_20_lower = calculate_donchian(high, low, 20)
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr_12h = np.zeros(n)
    tr_12h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_12h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_12h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 100  # Warmup for 1d chop and 12h indicators stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(chop_1d_aligned[i]) or np.isnan(donch_20_upper[i]) or np.isnan(donch_20_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- 1d Regime Filter: CHOP < 38.2 = trending, CHOP > 61.8 = ranging ---
        is_trending = chop_1d_aligned[i] < 38.2
        is_ranging = chop_1d_aligned[i] > 61.8
        # Neutral zone (38.2-61.8): no trades to avoid whipsaws
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Price ---
        price = close[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Exit on Donchian opposite breakout (trailing stop)
            if position_side > 0 and price < donch_20_lower[i]:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            elif position_side < 0 and price > donch_20_upper[i]:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Only trade in trending regime with volume confirmation
        if is_trending and volume_spike:
            # Long: Price breaks above Donchian upper channel
            if price > donch_20_upper[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Price breaks below Donchian lower channel
            elif price < donch_20_lower[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            # No signal in ranging/neutral regime or without volume confirmation
            signals[i] = 0.0
    
    return signals