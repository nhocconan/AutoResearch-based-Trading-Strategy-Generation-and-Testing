#!/usr/bin/env python3
"""
Experiment #5278: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation + ATR stoploss
HYPOTHESIS: On daily timeframe, Donchian channel breakouts capture strong momentum moves. 
Filtered by weekly HMA(21) trend direction to avoid counter-trend trades. 
Volume confirmation ensures breakout validity. 
ATR-based stoploss manages risk. 
Designed for 15-30 trades/year on 1d timeframe (60-120 total over 4 years) to minimize fee drag.
Works in bull markets by catching breakouts above weekly uptrend and in bear markets by catching breakdowns below weekly downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5278_1d_donchian20_1w_hma_vol_v1"
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
        # HMA formula: WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        wma_half = pd.Series(close_1w).ewm(span=half_len, adjust=False).mean().values
        wma_full = pd.Series(close_1w).ewm(span=21, adjust=False).mean().values
        hma_raw = 2 * wma_half - wma_full
        hma_21 = pd.Series(hma_raw).ewm(span=sqrt_len, adjust=False).mean().values
        hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Donchian Channel (20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 1d Indicators: ATR(14) for stoploss and volume filter ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    stop_price = 0.0
    
    warmup = max(20, 14, 20, 21)  # Donchian, ATR, volume, HMA warmup
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i]) or np.isnan(hma_21_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Update stoploss for existing position
        if in_position:
            if position_side == 1:  # Long
                # Trail stop: max(entry_price, highest_high - 2*ATR) or break even
                if i == warmup:  # First bar of position
                    stop_price = entry_price - 2.0 * atr[i]
                else:
                    stop_price = max(stop_price, entry_price)  # Break even stop
                    stop_price = max(stop_price, high[i] - 2.0 * atr[i])  # Trailing
                
                # Exit conditions: stoploss hit OR Donchian break down OR weekly trend change
                if price <= stop_price or price < donchian_low[i] or (hma_21_aligned[i] < hma_21_aligned[i-1] and position_side == 1):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                # Trail stop: min(entry_price, lowest_low + 2*ATR) or break even
                if i == warmup:  # First bar of position
                    stop_price = entry_price + 2.0 * atr[i]
                else:
                    stop_price = min(stop_price, entry_price)  # Break even stop
                    stop_price = min(stop_price, low[i] + 2.0 * atr[i])  # Trailing
                
                # Exit conditions: stoploss hit OR Donchian break up OR weekly trend change
                if price >= stop_price or price > donchian_high[i] or (hma_21_aligned[i] > hma_21_aligned[i-1] and position_side == -1):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry ---
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirm = vol > 1.5 * vol_ma[i]
        
        # Weekly trend filter: HMA rising/falling
        weekly_uptrend = hma_21_aligned[i] > hma_21_aligned[i-1]
        weekly_downtrend = hma_21_aligned[i] < hma_21_aligned[i-1]
        
        # Donchian breakout conditions
        breakout_up = price > donchian_high[i]
        breakout_down = price < donchian_low[i]
        
        # Entry logic
        if breakout_up and weekly_uptrend and vol_confirm:
            in_position = True
            position_side = 1
            entry_price = price
            stop_price = entry_price - 2.0 * atr[i]
            signals[i] = SIZE
        elif breakout_down and weekly_downtrend and vol_confirm:
            in_position = True
            position_side = -1
            entry_price = price
            stop_price = entry_price + 2.0 * atr[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals