#!/usr/bin/env python3
"""
Experiment #3543: 4h Donchian Breakout + 12h HMA Trend + Volume Confirmation
HYPOTHESIS: 4h Donchian(20) breakouts aligned with 12h HMA(21) trend direction and volume confirmation capture medium-term momentum with low overtrading. 
The 12h HMA provides a smoother trend filter than EMA, reducing whipsaws in ranging markets. Volume confirms breakout strength. 
Position size 0.25. Target: 75-200 total trades over 4 years (19-50/year).
Uses 12h for trend filter and 4h for entry timing and risk management.
Works in bull (continuation from uptrend) and bear (continuation from downtrend) via price channels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3543_4h_donchian20_12h_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 12h data for HMA trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate HMA(21) on 12h close
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    n_hma = 21
    half_n = n_hma // 2
    sqrt_n = int(np.sqrt(n_hma))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # Calculate WMA for 12h series
    wma_half = np.array([wma(close_12h[i:i+half_n], half_n) if i+half_n <= len(close_12h) else np.nan 
                         for i in range(len(close_12h))])
    wma_full = np.array([wma(close_12h[i:i+n_hma], n_hma) if i+n_hma <= len(close_12h) else np.nan 
                         for i in range(len(close_12h))])
    
    # HMA = WMA(2*WMA(half_n) - WMA(full_n)), sqrt_n)
    hma_12h_raw = 2 * wma_half - wma_full
    hma_12h = np.array([wma(hma_12h_raw[i:i+sqrt_n], sqrt_n) if i+sqrt_n <= len(hma_12h_raw) else np.nan 
                        for i in range(len(hma_12h_raw))])
    
    # Align HMA to 4h timeframe
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # === 4h Indicators: Donchian channels (20-period) for entry timing ===
    lookback_4h = 20
    highest_high_4h = pd.Series(high).rolling(window=lookback_4h, min_periods=lookback_4h).max().values
    lowest_low_4h = pd.Series(low).rolling(window=lookback_4h, min_periods=lookback_4h).min().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(50, lookback_4h, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high_4h[i]) or np.isnan(lowest_low_4h[i]) or
            np.isnan(hma_12h_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below Donchian low - trend reversal
                elif price < lowest_low_4h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above Donchian high - trend reversal
                elif price > highest_high_4h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.8x average) for confirmation
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Determine trend direction from 12h HMA
            # Uptrend: price above HMA, Downtrend: price below HMA
            price_vs_hma = price - hma_12h_aligned[i]
            
            # Long entry: price breaks above 4h Donchian high with bullish trend (above HMA)
            if (price > highest_high_4h[i] and 
                price_vs_hma > 0):  # Above 12h HMA = bullish trend
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below 4h Donchian low with bearish trend (below HMA)
            elif (price < lowest_low_4h[i] and 
                  price_vs_hma < 0):  # Below 12h HMA = bearish trend
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals