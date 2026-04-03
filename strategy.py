#!/usr/bin/env python3
"""
Experiment #078: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation + ATR stoploss

HYPOTHESIS: Donchian channel breakouts on daily timeframe capture strong momentum moves, 
filtered by 1-week HMA trend direction and volume confirmation. This structure works in both 
bull and bear markets by only taking breakouts in the direction of the higher timeframe trend. 
ATR-based stoploss manages risk. Targets 30-100 trades over 4 years (7-25/year) to minimize 
fee drag while capturing high-probability trend continuation moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_1d_htf_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HMA(21) on 1w close
    if len(df_1w) >= 21:
        close_1w = df_1w['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights / weights.sum(), mode='valid')
        
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, 21)
        wma_2x_sub = 2 * wma_half - wma_full
        hma_1w = wma(wma_2x_sub, sqrt_len)
        
        # Align to 1d timeframe
        hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    else:
        hma_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators ===
    # Donchian(20) - upper and lower bands
    donchian_period = 20
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    
    for i in range(donchian_period - 1, n):
        upper_band[i] = np.max(high[i - donchian_period + 1:i + 1])
        lower_band[i] = np.min(low[i - donchian_period + 1:i + 1])
    
    # Volume ratio (current vs 20-period average)
    vol_ratio = np.full(n, np.nan)
    if n >= 20:
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_ratio[20:] = volume[20:] / vol_ma[20:]
        vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # ATR(14) for stoploss
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr[14:] = pd.Series(tr[14:]).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    max_favorable_price = 0.0  # For trailing stop
    
    warmup = max(100, donchian_period, 20)  # Ensure enough data
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(hma_1w_aligned[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss with trailing) ---
        if in_position:
            # Update max favorable price
            if position_side > 0:  # Long
                max_favorable_price = max(max_favorable_price, high[i])
                stop_level = max_favorable_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                max_favorable_price = min(max_favorable_price, low[i])
                stop_level = max_favorable_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Trend filter: price > HMA for long, price < HMA for short
        price_above_hma = close[i] > hma_1w_aligned[i]
        price_below_hma = close[i] < hma_1w_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        # Long: Donchian breakout above upper band with volume and trend alignment
        long_condition = (
            close[i] > upper_band[i] and 
            price_above_hma and 
            volume_spike
        )
        
        # Short: Donchian breakdown below lower band with volume and trend alignment
        short_condition = (
            close[i] < lower_band[i] and 
            price_below_hma and 
            volume_spike
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