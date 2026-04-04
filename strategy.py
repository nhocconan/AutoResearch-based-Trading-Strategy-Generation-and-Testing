#!/usr/bin/env python3
"""
Experiment #3526: 4h Donchian Breakout + 1d Trend + Volume Confirmation
HYPOTHESIS: 4h Donchian(20) breakouts aligned with 1d EMA(50) trend and volume confirmation capture medium-term momentum.
The 1d EMA(50) provides a robust trend filter that works in both bull and bear markets - price above EMA(50) favors longs, below favors shorts.
Volume confirmation ensures breakouts have conviction. Position size 0.25. Target: 75-200 total trades over 4 years (19-50/year).
Uses 1d for trend filter and 4h only for entry timing and risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3526_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d timeframe
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align EMA(50) to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h Indicators: Donchian channels (20-period) for entry timing ===
    lookback_4h = 20
    highest_high_4h = pd.Series(high).rolling(window=lookback_4h, min_periods=lookback_4h).max().values
    lowest_low_4h = pd.Series(low).rolling(window=lookback_4h, min_periods=lookback_4h).min().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    warmup = max(50, lookback_4h, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high_4h[i]) or np.isnan(lowest_low_4h[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Stoploss Logic ---
        if signals[i-1] > 0:  # Long position
            if price < high[i] - 2.5 * atr[i]:  # Stoploss: 2.5*ATR below session high
                signals[i] = 0.0
            else:
                # Continue long if price above 1d EMA(50) and volume confirmation
                if price > ema_50_1d_aligned[i] and vol_ratio[i] > 1.5:
                    signals[i] = SIZE
                else:
                    signals[i] = 0.0
        elif signals[i-1] < 0:  # Short position
            if price > low[i] + 2.5 * atr[i]:  # Stoploss: 2.5*ATR above session low
                signals[i] = 0.0
            else:
                # Continue short if price below 1d EMA(50) and volume confirmation
                if price < ema_50_1d_aligned[i] and vol_ratio[i] > 1.5:
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
        else:
            # --- New Position Entry Logic ---
            # Require volume confirmation (> 1.5x average) for entry
            volume_confirm = vol_ratio[i] > 1.5
            
            if volume_confirm:
                # Long entry: price breaks above 4h Donchian high with bullish trend (above 1d EMA50)
                if (price > highest_high_4h[i] and 
                    price > ema_50_1d_aligned[i]):
                    signals[i] = SIZE
                # Short entry: price breaks below 4h Donchian low with bearish trend (below 1d EMA50)
                elif (price < lowest_low_4h[i] and 
                      price < ema_50_1d_aligned[i]):
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals