#!/usr/bin/env python3
"""
Experiment #5073: 4h Donchian(20) Breakout + 12h HMA Trend + Volume Spike + ATR Stoploss
HYPOTHESIS: On 4h timeframe, Donchian(20) breakouts aligned with 12h HMA trend capture strong momentum with controlled frequency. The 12h HMA acts as a robust trend filter (superior to EMA in choppy markets). Volume > 1.5x average confirms participation. ATR(14) trailing stop (2.0x) manages risk. Designed for 19-50 trades/year on 4h timeframe to minimize fee drag while maintaining statistical significance. Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5073_4h_donchian20_12h_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 12h data for HMA trend
    df_12h = get_htf_data(prices, '12h')
    
    # === 12h Indicators: HMA(21) for trend ===
    if len(df_12h) >= 21:
        # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = len(df_12h) // 2
        sqrt_len = int(np.sqrt(len(df_12h)))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights, mode='valid') / weights.sum()
        
        close_12h = df_12h['close'].values
        wma_half = wma(close_12h, half_len)
        wma_full = wma(close_12h, len(close_12h))
        # For simplicity, use EMA approximation of HMA: EMA(WMA(2*WMA(n/2)-WMA(n)))
        # We'll use a simpler but effective trend: HMA ≈ 2*WMA(n/2) - WMA(n) then WMA of that
        if len(wma_half) >= half_len and len(wma_full) >= len(close_12h):
            # Align lengths by taking last values
            raw_hma = 2 * wma_half[-half_len:] - wma_full[-len(close_12h):]
            if len(raw_hma) >= sqrt_len:
                hma_values = wma(raw_hma, sqrt_len)
                # Pad to match original length
                hma_12h = np.full(len(close_12h), np.nan)
                hma_12h[-len(hma_values):] = hma_values
            else:
                hma_12h = np.full(len(close_12h), np.nan)
        else:
            hma_12h = np.full(len(close_12h), np.nan)
        
        # Fallback to simple EMA if HMA calculation fails
        if np.all(np.isnan(hma_12h)):
            hma_12h = pd.Series(close_12h).ewm(span=21, min_periods=21, adjust=False).mean().values
    else:
        hma_12h = np.full(len(df_12h), np.nan)
    
    # Align HMA to 4h timeframe
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # === 4h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 21)  # Donchian, Volume MA, ATR, HMA warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(hma_12h_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions with 12h HMA trend filter
        # Long: Donchian breakout above + price above 12h HMA (uptrend)
        # Short: Donchian breakdown below + price below 12h HMA (downtrend)
        breakout_long = (price >= high_roll[i]) and (price > hma_12h_aligned[i]) and vol_confirm
        breakout_short = (price <= low_roll[i]) and (price < hma_12h_aligned[i]) and vol_confirm
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>