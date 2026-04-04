#!/usr/bin/env python3
"""
Experiment #5269: 4h Donchian(20) breakout + HMA(21) trend + volume confirmation + ATR stoploss
HYPOTHESIS: On 4h timeframe, Donchian(20) breakouts capture strong momentum moves, filtered by HMA(21) trend direction and volume > 1.5x average to avoid false breakouts. ATR-based stoploss (2.5x) manages risk. Designed for 20-50 trades/year on 4h timeframe (80-200 total over 4 years) to minimize fee drag. Works in bull markets by catching upward breakouts and in bear markets by catching downward breakouts, while avoiding ranging conditions where price stays within Donchian channels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5269_4h_donchian20_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for regime filter (optional, can be removed if too restrictive) ===
    # Not using HTF for now to keep it simple and avoid over-filtering
    
    # === 4h Indicators: Donchian Channel (20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: HMA(21) for trend filter ===
    def hma(arr, period):
        half = int(period / 2)
        sqrt = int(np.sqrt(period))
        wma2 = pd.Series(arr).ewm(span=half, adjust=False).mean().values
        wma1 = pd.Series(arr).ewm(span=period, adjust=False).mean().values
        raw = 2 * wma2 - wma1
        hma_vals = pd.Series(raw).ewm(span=sqrt, adjust=False).mean().values
        return hma_vals
    
    hma_21 = hma(close, 21)
    
    # === 4h Indicators: ATR(14) for stoploss and volume average ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    stop_price = 0.0
    
    warmup = max(20, 21, 14, 20)  # Donchian, HMA, ATR, volume MA warmup
    
    for i in range(warmup, n):
        # --- Session Filter: 00-24 UTC (4h timeframe, already captures major sessions) ---
        # 4h candles already filter to 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC
        # These cover major session opens, so no additional filter needed
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_21[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # --- Stoploss Check ---
        if in_position:
            if position_side > 0:  # Long position
                if price <= stop_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                else:
                    signals[i] = SIZE
            else:  # Short position
                if price >= stop_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Donchian breakout conditions
        breakout_up = price > donchian_high[i-1]  # Break above previous period's high
        breakout_down = price < donchian_low[i-1]  # Break below previous period's low
        
        # HMA trend filter
        hma_bullish = price > hma_21[i]
        hma_bearish = price < hma_21[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = vol > 1.5 * vol_ma[i]
        
        # Entry conditions
        if breakout_up and hma_bullish and vol_confirm:
            in_position = True
            position_side = 1
            entry_price = price
            stop_price = entry_price - 2.5 * atr[i]  # 2.5x ATR stoploss
            signals[i] = SIZE
        elif breakout_down and hma_bearish and vol_confirm:
            in_position = True
            position_side = -1
            entry_price = price
            stop_price = entry_price + 2.5 * atr[i]  # 2.5x ATR stoploss
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals