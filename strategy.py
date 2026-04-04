#!/usr/bin/env python3
"""
Experiment #4343: 4h Donchian(20) Breakout + 12h HMA Trend + Volume Spike
HYPOTHESIS: Donchian channel breakouts capture strong momentum moves. 12h HMA(21) filters direction (only trade long when price > HMA, short when price < HMA). Volume > 2.0x average confirms institutional participation. ATR(14) trailing stop (2.5x) manages risk. Designed for 4h timeframe to avoid overtrading - target 75-200 total trades over 4 years (19-50/year). Position size 0.25 balances profit potential with drawdown control in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4343_4h_donchian20_12h_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === Precompute HTF: 12h HMA(21) for trend filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 21:
        # Hull Moving Average: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        def wma(data, window):
            weights = np.arange(1, window + 1)
            return np.convolve(data, weights / weights.sum(), mode='valid')
        
        close_12h = df_12h['close'].values
        n_12h = len(close_12h)
        half = 21 // 2
        sqrt_n = int(np.sqrt(21))
        
        if n_12h >= 21:
            wma_full = np.convolve(close_12h, np.arange(1, 22) / np.arange(1, 22).sum(), mode='valid')
            wma_half = np.convolve(close_12h, np.arange(1, half + 1) / np.arange(1, half + 1).sum(), mode='valid')
            hma_12h = np.convolve(2 * wma_half - wma_full, np.arange(1, sqrt_n + 1) / np.arange(1, sqrt_n + 1).sum(), mode='valid')
            
            # Pad to match original length
            hma_padded = np.full(n_12h, np.nan)
            start_idx = 21 - 1  # WMA(21) needs 21 points
            hma_padded[start_idx:start_idx + len(hma_12h)] = hma_12h
            hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_padded)
        else:
            hma_12h_aligned = np.full(n, np.nan)
    else:
        hma_12h_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel(20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14)  # Donchian, vol MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(hma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 2.0x average) to filter noise
        volume_confirm = vol_ratio[i] > 2.0
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i-1]  # Close above previous period's high
        breakout_down = close[i] < donchian_low[i-1]  # Close below previous period's low
        
        # Trend filter: price relative to 12h HMA
        price_above_hma = price > hma_12h_aligned[i]
        price_below_hma = price < hma_12h_aligned[i]
        
        # Long conditions: Donchian breakout up + volume + price above HMA
        long_entry = breakout_up and volume_confirm and price_above_hma
        
        # Short conditions: Donchian breakout down + volume + price below HMA
        short_entry = breakout_down and volume_confirm and price_below_hma
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals