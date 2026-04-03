#!/usr/bin/env python3
"""
Experiment #154: 1h RSI(2) mean reversion + 4h trend filter + volume confirmation + session filter
HYPOTHESIS: In 1h timeframe, RSI(2) < 10 indicates oversold bounce opportunity in uptrend (4h EMA50 > EMA200), 
RSI(2) > 90 indicates overbought reversal in downtrend (4h EMA50 < EMA200). Volume > 1.5x average confirms momentum. 
Session filter (08-20 UTC) reduces noise. Discrete sizing 0.20 manages risk. Target: 90-180 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_154_1h_rsi2_4htrend_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Pre-compute session hours for efficiency
    hours = prices.index.hour
    
    # === HTF: 4h data for trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA50 and EMA200 for trend direction
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200_4h = pd.Series(close_4h).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align to 1h timeframe (automatically shifted by 1 for completed bars only)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # === 1h Indicators: RSI(2) for mean reversion signals ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = pd.Series(gain).ewm(alpha=1/2, min_periods=2, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, min_periods=2, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_2 = 100 - (100 / (1 + rs))
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 200  # Need enough bars for 4h EMA200
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(ema200_4h_aligned[i]) or
            np.isnan(rsi_2[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Trend Filter from 4h: Uptrend if EMA50 > EMA200, Downtrend if EMA50 < EMA200 ---
        uptrend_4h = ema50_4h_aligned[i] > ema200_4h_aligned[i]
        downtrend_4h = ema50_4h_aligned[i] < ema200_4h_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit at RSI(2) > 80 (exit overextended longs)
                if rsi_2[i] > 80:
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
                # Take profit at RSI(2) < 20 (exit overextended shorts)
                if rsi_2[i] < 20:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period: 2 bars
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Oversold bounce in uptrend
        if (rsi_2[i] < 10 and uptrend_4h and volume_spike):
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Overbought reversal in downtrend
        elif (rsi_2[i] > 90 and downtrend_4h and volume_spike):
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals