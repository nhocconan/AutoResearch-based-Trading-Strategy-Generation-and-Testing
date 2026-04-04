#!/usr/bin/env python3
"""
Experiment #5278: 1d Donchian(20) breakout + HMA(21) trend + volume confirmation + ATR stoploss
HYPOTHESIS: On daily timeframe, Donchian channel breakouts capture strong momentum moves. 
Filtered by 1-week HMA trend to ensure alignment with higher timeframe direction, and volume 
confirmation to avoid false breakouts. Uses ATR-based stoploss (signal→0 when price moves 
2*ATR against position) to manage risk. Designed for 15-25 trades/year on 1d timeframe 
(60-100 total over 4 years) to minimize fee drag. Works in bull markets by catching 
breakouts above upper channel and in bear markets by catching breakdowns below lower channel.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5278_1d_donchian20_hma21_vol_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 21:
        # Calculate HMA(21) on weekly close
        close_1w = df_1w['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        wma_half = pd.Series(close_1w).ewm(span=half_len, adjust=False).mean().values
        wma_full = pd.Series(close_1w).ewm(span=21, adjust=False).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_21 = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False).mean().values
        hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Donchian Channel (20) ===
    donchian_window = 20
    upper_channel = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().shift(1).values
    lower_channel = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().shift(1).values
    
    # === 1d Indicators: Volume confirmation (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    # === 1d Indicators: ATR(14) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    max_favorable_price = 0.0  # For trailing stop logic if needed
    
    warmup = max(donchian_window, 21, 20, 14)  # Donchian, HMA, volume, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # --- Stoploss Logic: Close when price moves 2*ATR against position ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr[i]
                if price < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                stop_level = entry_price + 2.0 * atr[i]
                if price > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Donchian breakout conditions
        breakout_up = price > upper_channel[i]
        breakout_down = price < lower_channel[i]
        
        # HMA trend filter (1-week)
        hma_bullish = price > hma_21_aligned[i]
        hma_bearish = price < hma_21_aligned[i]
        
        # Volume confirmation (above average)
        volume_confirm = vol > vol_ma[i]
        
        # Entry conditions
        if breakout_up and hma_bullish and volume_confirm:
            in_position = True
            position_side = 1
            entry_price = price
            max_favorable_price = price
            signals[i] = SIZE
        elif breakout_down and hma_bearish and volume_confirm:
            in_position = True
            position_side = -1
            entry_price = price
            max_favorable_price = price
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals