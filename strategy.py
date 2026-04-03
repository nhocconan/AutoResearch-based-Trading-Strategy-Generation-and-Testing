#!/usr/bin/env python3
"""
Experiment #228: 12h Donchian Breakout + Volume Spike + ATR Stoploss

HYPOTHESIS: Donchian channel breakouts on 12h timeframe capture significant trends in BTC/ETH/SOL.
Price breaking above/below 20-period Donchian channel with volume confirmation (>1.5x average)
indicates institutional participation. ATR-based stoploss (2.5x ATR) limits downside during
false breakouts. 12h timeframe targets 12-37 trades/year (50-150 total over 4 years) to
minimize fee drag while capturing major moves. Works in bull markets via trend continuation
and bear markets via failed reversals at channel extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_228_12h_donchian_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for Donchian channel and trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 20-period Donchian channel on 1w
    def calculate_donchian(high_arr, low_arr, period=20):
        """Calculate Donchian channel: upper, lower, middle"""
        upper = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        middle = (upper + lower) / 2.0
        return upper, lower, middle
    
    donchian_upper_1w, donchian_lower_1w, donchian_middle_1w = calculate_donchian(
        df_1w['high'].values, df_1w['low'].values
    )
    
    # Align Donchian levels to 12h timeframe
    donchian_upper_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_1w)
    donchian_lower_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_1w)
    donchian_middle_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_middle_1w)
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
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
    
    warmup = 100  # Warmup for 1w indicators stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper_1w_aligned[i]) or np.isnan(donchian_lower_1w_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Price Levels ---
        price = close[i]
        upper = donchian_upper_1w_aligned[i]
        lower = donchian_lower_1w_aligned[i]
        middle = donchian_middle_1w_aligned[i]
        ema50 = ema50_1w_aligned[i]
        
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
            
            # Exit on mean reversion to middle channel
            if abs(price - middle) < 0.5 * atr_14[i]:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Volume confirmation: Require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        # Long breakout: Price > upper channel + volume spike + price > EMA50 (uptrend)
        long_breakout = (price > upper) and volume_spike and (price > ema50)
        
        # Short breakout: Price < lower channel + volume spike + price < EMA50 (downtrend)
        short_breakout = (price < lower) and volume_spike and (price < ema50)
        
        if long_breakout:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_breakout:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals