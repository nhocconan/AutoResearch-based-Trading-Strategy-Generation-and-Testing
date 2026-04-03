#!/usr/bin/env python3
"""
Experiment #244: 1d Donchian20 + 1w Regime + Volume Strategy

HYPOTHESIS: Donchian(20) breakout on 1d combined with 1w trend filter (price above/below EMA50) and volume confirmation captures strong directional moves while avoiding choppy markets. In trending regimes (price above EMA50 for longs, below EMA50 for shorts), we trade breakouts with the higher timeframe trend. Uses ATR-based stoploss and minimum 3-day holding period to reduce churn. Target: 50-120 trades over 4 years (12-30/year) to stay within fee drag limits for 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_244_1d_donchian20_1w_regime_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for regime detection (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for regime filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === 1d Indicators: Donchian(20) channels ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # === 1d Indicators: ATR(14) for stoploss ===
    tr_1d = np.zeros(n)
    tr_1d[0] = high[0] - low[0]
    for i in range(1, n):
        tr_1d[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 1d Indicators: Volume MA(20) for spike detection ===
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
    days_since_entry = 0  # Track days in position for minimum holding period
    
    warmup = 200  # Warmup for 1d EMA200 equivalent stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema50_1w_aligned[i]) or
            np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            days_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    days_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on opposite Donchian breakout (contrarian exit)
                if low[i] < donch_lower[i-1] and vol_ratio[i] > 1.8:
                    in_position = False
                    position_side = 0
                    days_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    days_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on opposite Donchian breakout (contrarian exit)
                if high[i] > donch_upper[i-1] and vol_ratio[i] > 1.8:
                    in_position = False
                    position_side = 0
                    days_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 3 days to reduce churn
            if days_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # --- 1w Regime Filter: Only trade when price is aligned with 1w trend ---
        # Long: price above 1w EMA50 (bullish regime)
        # Short: price below 1w EMA50 (bearish regime)
        is_bullish_regime = price > ema50_1w_aligned[i]
        is_bearish_regime = price < ema50_1w_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Donchian Breakout Conditions ---
        breakout_up = high[i] > donch_upper[i-1]  # Break above upper channel
        breakout_down = low[i] < donch_lower[i-1]  # Break below lower channel
        
        # --- Entry Conditions ---
        # Long: bullish regime + Donchian breakout up + volume spike
        if is_bullish_regime and breakout_up and volume_spike:
            in_position = True
            position_side = 1
            entry_price = close[i]
            days_since_entry = 0
            signals[i] = SIZE
        # Short: bearish regime + Donchian breakout down + volume spike
        elif is_bearish_regime and breakout_down and volume_spike:
            in_position = True
            position_side = -1
            entry_price = close[i]
            days_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals