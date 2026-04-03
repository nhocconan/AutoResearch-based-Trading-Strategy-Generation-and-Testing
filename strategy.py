#!/usr/bin/env python3
"""
Experiment #037: 4h Donchian(20) Breakout + HMA Trend + Volume Spike + ATR Stoploss

HYPOTHESIS: Donchian(20) breakouts on 4h capture institutional participation in trends.
Confirmed by 4h HMA(21/55) trend alignment and 1d volume spike (>2.0x average).
Uses discrete position sizing (0.30) and ATR(14) stoploss (2.5x) to manage drawdown.
HTF: 1d for volume confirmation, 1w for regime filter (optional). Target: 75-200 trades over 4 years.
Works in both bull (breakout continuation) and bear (breakdown continuation) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 1d data for volume spike confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === 4h Indicators: Donchian(20) and HMA(21,55) ===
    # Donchian(20) channels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    if n >= 20:
        for i in range(20, n):
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # HMA(21) and HMA(55) for trend
    def hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = pd.Series(arr).ewm(span=half, adjust=False).mean()
        wma1 = pd.Series(arr).ewm(span=period, adjust=False).mean()
        raw = 2 * wma2 - wma1
        hma_val = pd.Series(raw).ewm(span=sqrt, adjust=False).mean()
        return hma_val.values
    
    hma_21 = hma(close, 21)
    hma_55 = hma(close, 55)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position sizing (30% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    max_favorable_price = 0.0  # For trailing stop logic
    
    warmup = max(100, 55)  # Ensure enough data for HMA(55)
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_21[i]) or np.isnan(hma_55[i]) or 
            np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: HMA21 > HMA55 for uptrend, < for downtrend ---
        hma_uptrend = hma_21[i] > hma_55[i]
        hma_downtrend = hma_21[i] < hma_55[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # --- Exit Logic (Trailing stop based on ATR) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                # Update max favorable price
                max_favorable_price = max(max_favorable_price, high[i])
                # Trailing stop: 2.5 * ATR below max favorable price
                stop_level = max_favorable_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    max_favorable_price = 0.0
                    continue
            else:  # Short position
                # Update max favorable price (lowest for shorts)
                if max_favorable_price == 0.0:
                    max_favorable_price = low[i]
                else:
                    max_favorable_price = min(max_favorable_price, low[i])
                # Trailing stop: 2.5 * ATR above max favorable price
                stop_level = max_favorable_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    max_favorable_price = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long when price breaks above Donchian High with volume and HMA uptrend
        long_condition = (
            close[i] > donchian_high[i] and 
            volume_spike and 
            hma_uptrend
        )
        
        # Short when price breaks below Donchian Low with volume and HMA downtrend
        short_condition = (
            close[i] < donchian_low[i] and 
            volume_spike and 
            hma_downtrend
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            max_favorable_price = high[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            max_favorable_price = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals