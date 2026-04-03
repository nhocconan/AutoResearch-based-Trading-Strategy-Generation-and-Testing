#!/usr/bin/env python3
"""
Experiment #1928: 12h Donchian(20) Breakout + 1w EMA Trend + Volume Confirmation
HYPOTHESIS: 12h Donchian breakouts aligned with 1-week EMA trend and volume spikes capture institutional momentum with low trade frequency. Works in bull/bear by following higher timeframe trend. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1928_12h_donchian20_1w_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for EMA trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA(50) trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1w = np.where(close_1w > ema_50_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 12h Indicators: Donchian(20) channels ===
    # Highest high of last 20 bars (including current)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lowest low of last 20 bars (including current)
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Donchian breakout levels (using previous bar's channel to avoid look-ahead)
    donchian_high = np.roll(highest_high, 1)
    donchian_low = np.roll(lowest_low, 1)
    donchian_high[0] = np.nan  # First bar has no previous channel
    donchian_low[0] = np.nan
    
    # Volume MA(20) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for EMA(50) and Donchian(20)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss and Donchian opposite exit ---
        if in_position:
            bars_since_entry += 1
            
            # Calculate ATR(14) for dynamic stoploss
            if i >= 14:
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                tr = max(tr1, tr2, tr3)
                # Simple ATR approximation using recent TR (for speed)
                atr_approx = np.mean([
                    max(high[i-14+i] - low[i-14+i], 
                        abs(high[i-14+i] - close[i-14+i-1]) if i-14+i > 0 else 0,
                        abs(low[i-14+i] - close[i-14+i-1]) if i-14+i > 0 else 0)
                    for j in range(14) if i-14+j >= 0 and i-14+j < i
                ]) if i >= 28 else 0.01 * price  # fallback
                
                # Ensure we have a reasonable ATR value
                if atr_approx <= 0:
                    atr_approx = 0.01 * price
            else:
                atr_approx = 0.01 * price  # 1% of price as fallback
            
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5 * ATR below entry
                if price < entry_price - 2.5 * atr_approx:
                    exit_signal = True
                # Exit if price breaks below Donchian low (opposite channel)
                elif price < donchian_low[i]:
                    exit_signal = True
            else:  # Short position
                # Stoploss: 2.5 * ATR above entry
                if price > entry_price + 2.5 * atr_approx:
                    exit_signal = True
                # Exit if price breaks above Donchian high (opposite channel)
                elif price > donchian_high[i]:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1w trend alignment for bias filter
        trend_bias = trend_1w_aligned[i]
        
        # Volume confirmation: require volume spike (> 2.0x average for stricter filter)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long entry: price breaks above Donchian high AND 1w trend up
            if trend_bias > 0 and price > donchian_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low AND 1w trend down
            elif trend_bias < 0 and price < donchian_low[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals