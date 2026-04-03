#!/usr/bin/env python3
"""
Experiment #074: 1h Donchian(20) Breakout + 4h/1d HTF Direction + Volume Spike + Session Filter

HYPOTHESIS: Donchian channel breakouts on 1h timeframe, filtered by 4h trend (price > 4h EMA21 = bullish, price < 4h EMA21 = bearish) 
and 1d bias (close > 1d EMA50 = bullish, close < 1d EMA50 = bearish), volume spikes (>1.8x average), and session filter (08-20 UTC) 
capture strong momentum moves with reduced false breakouts. HTF (4h/1d) provides signal direction, 1h only for entry timing. 
Position size fixed at 0.20 to manage drawdown. ATR-based stoploss (2.0x) manages risk. Targets 60-150 total trades over 4 years 
(15-37/year) to minimize fee drag while capturing significant moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_074_1h_donchian_4h_1d_direction_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === Pre-compute session hours (08-20 UTC) ===
    # prices.index is DatetimeIndex, .hour works directly
    hours = prices.index.hour
    
    # === HTF: 4h data for EMA21 trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) >= 21:
        ema_4h = pd.Series(df_4h['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
        ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)  # auto shift(1)
    else:
        ema_4h_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for EMA50 bias (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)  # auto shift(1)
    else:
        ema_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Donchian Channel (20) ===
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    for i in range(20, n):
        donchian_h[i] = np.max(high[i-20:i])
        donchian_l[i] = np.min(low[i-20:i])
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)  # Default to 1.0 (neutral)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Fixed position size (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position
    
    warmup = 50  # Ensure enough data for HTF EMA, ATR, Donchian
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- HTF Direction Filters ---
        # 4h trend: price > 4h EMA21 = bullish, price < 4h EMA21 = bearish
        bullish_4h = close[i] > ema_4h_aligned[i]
        bearish_4h = close[i] < ema_4h_aligned[i]
        
        # 1d bias: close > 1d EMA50 = bullish, close < 1d EMA50 = bearish
        bullish_1d = close[i] > ema_1d_aligned[i]
        bearish_1d = close[i] < ema_1d_aligned[i]
        
        # Combined HTF direction (both must agree)
        htf_bullish = bullish_4h and bullish_1d
        htf_bearish = bearish_4h and bearish_1d
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Donchian Breakout Conditions ---
        breakout_up = close[i] > donchian_h[i]
        breakout_down = close[i] < donchian_l[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout up + volume spike + HTF bullish
        long_condition = breakout_up and volume_spike and htf_bullish
        
        # Short: Donchian breakout down + volume spike + HTF bearish
        short_condition = breakout_down and volume_spike and htf_bearish
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals