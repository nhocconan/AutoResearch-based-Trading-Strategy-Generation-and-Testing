#!/usr/bin/env python3
"""
Experiment #2862: 12h Donchian(20) breakout + HMA trend + volume confirmation + ATR stoploss
HYPOTHESIS: 12h Donchian breakouts capture medium-term trends in BTC/ETH/SOL. 
HMA(21) on 1d timeframe filters trend direction, volume spike (>2.0x MA20) confirms 
momentum, and ATR-based stoploss manages risk. This structure worked well on SOLUSDT 
in previous experiments (test Sharpe 1.10-1.38). 12h timeframe targets 50-150 trades 
over 4 years (12-37/year) to minimize fee drag while maintaining statistical significance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2862_12h_donchian20_hma_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate HMA(21) on daily close
    def hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma2 = pd.Series(arr).ewm(span=half_period, adjust=False).mean().values
        wma1 = pd.Series(arr).ewm(span=period, adjust=False).mean().values
        raw = 2 * wma2 - wma1
        hma_vals = pd.Series(raw).ewm(span=sqrt_period, adjust=False).mean().values
        return hma_vals
    
    hma_1d = hma(close_1d, 21)
    trend_1d = np.where(close_1d > hma_1d, 1, -1)  # 1 = uptrend, -1 = downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 12h Indicators: Donchian channels (20) ===
    donchian_window = 20
    highest_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # ATR-based stoploss: exit if price moves 2.5*ATR against position
            if position_side > 0:  # Long
                if price < entry_price - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if price > entry_price + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) for confirmation
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Get daily trend bias
            trend_bias = trend_1d_aligned[i]
            
            # Long entry: price breaks above Donchian upper band in daily uptrend
            if trend_bias > 0 and price > highest_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower band in daily downtrend
            elif trend_bias < 0 and price < lowest_low[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals