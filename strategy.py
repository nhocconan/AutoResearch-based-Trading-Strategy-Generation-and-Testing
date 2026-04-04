#!/usr/bin/env python3
"""
Experiment #5270: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation
HYPOTHESIS: On daily timeframe, Donchian channel breakouts capture strong trending moves. 
Filtering with 1-week HMA ensures we only trade in the direction of the higher timeframe trend, 
avoiding counter-trend breakouts that fail. Volume confirmation adds conviction to breakouts. 
Designed for low frequency (15-30 trades/year) to minimize fee drag. Works in bull markets 
by catching upside breakouts and in bear markets by catching downside breakouts (via short signals).
Uses discrete position sizing (0.25) and ATR-based stoploss to control drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5270_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === Precompute HTF: 1w data for HMA trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 21:
        # Calculate HMA(21) on weekly close
        close_1w = df_1w['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights / weights.sum(), mode='valid')
        # Pad WMA results to align with original array
        wma_full = np.concatenate([np.full(half_len-1, np.nan), wma(close_1w, half_len)])
        wma_half = np.concatenate([np.full(half_len-1, np.nan), wma(close_1w, half_len)])
        wma_full_n = np.concatenate([np.full(21-1, np.nan), wma(close_1w, 21)])
        raw_hma = 2 * wma_half - wma_full_n
        hma_1w = np.concatenate([np.full(sqrt_len-1, np.nan), wma(raw_hma, sqrt_len)])
        hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    else:
        hma_1w_aligned = np.full(n, np.nan)
    
    # === Daily Indicators: Donchian(20) channels ===
    # Upper band: 20-period high
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Daily Indicators: Volume confirmation (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Daily Indicators: ATR(14) for stoploss ===
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0  # 1 for long, -1 for short
    entry_price = 0.0
    entry_bar = 0
    
    warmup = max(20, 20, 20, 14, 21)  # Donchian, volume, ATR, HMA warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(hma_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # --- Exit Logic: Stoploss or trend reversal ---
        if in_position:
            # Stoploss: 2 * ATR against position
            if position_side > 0:  # Long position
                if price < entry_price - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                if price > entry_price + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Donchian breakout conditions
        breakout_up = price > donch_high[i-1]  # Break above previous upper band
        breakout_down = price < donch_low[i-1]  # Break below previous lower band
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirm = vol > 1.5 * vol_ma[i]
        
        # 1w HMA trend filter: price above/below HMA
        price_above_hma = price > hma_1w_aligned[i]
        price_below_hma = price < hma_1w_aligned[i]
        
        # Entry conditions
        if breakout_up and vol_confirm and price_above_hma:
            in_position = True
            position_side = 1
            entry_price = price
            entry_bar = i
            signals[i] = SIZE
        elif breakout_down and vol_confirm and price_below_hma:
            in_position = True
            position_side = -1
            entry_price = price
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals