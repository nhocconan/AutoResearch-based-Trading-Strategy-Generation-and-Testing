#!/usr/bin/env python3
"""
Experiment #004: 1d Donchian(20) Breakout + 1w HMA Trend + Volume Spike + ATR Stoploss

HYPOTHESIS: Daily Donchian channel breakouts (20-period) aligned with weekly HMA(21) trend direction,
confirmed by 1d volume spike (>2x average), produce high-probability trades in both bull and bear markets.
The weekly trend filter prevents counter-trend trading during strong moves, while volume confirmation
ensures institutional participation. Targets 15-25 trades/year on 1d timeframe (60-100 total over 4 years)
to minimize fee drag while capturing significant breakouts. Uses discrete position sizing (0.30) and
ATR(14) stoploss (2.5x) to manage risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_1d_hma_1w_volume_v1"
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
        
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights/weights.sum(), mode='valid')
        
        if len(close_1w) >= 21:
            wma_half = wma(close_1w, half_len)
            wma_full = wma(close_1w, 21)
            wma_2x_sub = 2 * wma_half - wma_full
            hma_1w = wma(wma_2x_sub, sqrt_len)
            # Pad beginning with NaN
            hma_1w_padded = np.full(len(close_1w), np.nan)
            hma_1w_padded[half_len + sqrt_len - 1:] = hma_1w
            hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_padded)
        else:
            hma_1w_aligned = np.full(n, np.nan)
    else:
        hma_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators ===
    # Calculate Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate ATR(14) for stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Calculate 1d volume ratio (current vs 20-period average)
    vol_ratio_1d = np.full(n, 1.0)
    if n >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d[20:] = volume[20:] / vol_ma_20[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position sizing (30% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Align with weekly HMA direction ---
        # For simplicity, use price > HMA for uptrend bias, price < HMA for downtrend bias
        price_above_hma = close[i] > hma_1w_aligned[i]
        price_below_hma = close[i] < hma_1w_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_1d[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian high with volume and weekly uptrend bias
        long_condition = (
            close[i] > donchian_high[i] and 
            volume_spike and 
            price_above_hma
        )
        
        # Short: Price breaks below Donchian low with volume and weekly downtrend bias
        short_condition = (
            close[i] < donchian_low[i] and 
            volume_spike and 
            price_below_hma
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals