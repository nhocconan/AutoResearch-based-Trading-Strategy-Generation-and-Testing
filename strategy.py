#!/usr/bin/env python3
"""
Experiment #1873: 4h Donchian(20) breakout + HMA(21) trend + volume confirmation + ATR stoploss
HYPOTHESIS: Donchian channel breakouts capture strong momentum moves. HMA(21) filters for trend direction, volume confirmation (>1.5x average) ensures institutional participation, and ATR-based stoploss manages risk. Works in both bull and bear markets by following the 12h trend. Target: 75-200 total trades over 4 years (19-50/year) with discrete position sizing of 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1873_4h_donchian20_hma21_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h HMA(21) for trend direction
    def calculate_hma(series, period):
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = pd.Series(series).ewm(span=half_period, adjust=False).mean()
        wma_full = pd.Series(series).ewm(span=period, adjust=False).mean()
        raw_hma = 2 * wma_half - wma_full
        hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean()
        return hma.values
    
    hma_21_12h = calculate_hma(close_12h, 21)
    trend_12h = np.where(close_12h > hma_21_12h, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # === 4h Indicators: Donchian(20) channels ===
    donchian_period = 20
    upper_channel = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # === 4h Indicators: HMA(21) for entry filter ===
    hma_21 = calculate_hma(close, 21)
    
    # === 4h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    stop_price = 0.0
    
    warmup = 50  # sufficient for Donchian(20) and ATR(14)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(hma_21[i]) or np.isnan(trend_12h_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Stoploss or reverse signal ---
        if in_position:
            bars_since_entry += 1
            
            # Check stoploss
            if position_side > 0:  # Long position
                if price <= stop_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                if price >= stop_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Check for reverse signal
            long_signal = (price > upper_channel[i] and 
                          hma_21[i] > hma_21[i-1] and  # HMA rising
                          vol_ratio[i] > 1.5 and
                          trend_12h_aligned[i] > 0)
            
            short_signal = (price < lower_channel[i] and 
                           hma_21[i] < hma_21[i-1] and  # HMA falling
                           vol_ratio[i] > 1.5 and
                           trend_12h_aligned[i] < 0)
            
            if position_side > 0 and short_signal:
                in_position = False
                position_side = 0
                signals[i] = 0.0
            elif position_side < 0 and long_signal:
                in_position = False
                position_side = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        long_signal = (price > upper_channel[i] and 
                      hma_21[i] > hma_21[i-1] and  # HMA rising
                      vol_ratio[i] > 1.5 and
                      trend_12h_aligned[i] > 0)
        
        short_signal = (price < lower_channel[i] and 
                       hma_21[i] < hma_21[i-1] and  # HMA falling
                       vol_ratio[i] > 1.5 and
                       trend_12h_aligned[i] < 0)
        
        if long_signal:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            stop_price = entry_price - 2.5 * atr[i]  # 2.5 ATR stoploss
            signals[i] = SIZE
        elif short_signal:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            stop_price = entry_price + 2.5 * atr[i]  # 2.5 ATR stoploss
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals