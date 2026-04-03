#!/usr/bin/env python3
"""
Experiment #133: 4h Donchian Breakout + 12h HMA Trend + Volume Confirmation

HYPOTHESIS: Donchian(20) breakouts capture strong momentum moves. 
12h HMA(21) filter ensures we only trade in the direction of the higher timeframe trend.
Volume confirmation (>1.5x average volume) filters out weak breakouts.
ATR-based stoploss (2.5x ATR) manages risk. 
Works in bull markets by catching breakouts, in bear markets by catching breakdowns.
Target: 75-200 total trades over 4 years (19-50/year).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_12h_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for trend filter (HMA) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate HMA on 12h data
    def calculate_hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = pd.Series(arr).ewm(span=half_period, adjust=False).mean().values
        wma_full = pd.Series(arr).ewm(span=period, adjust=False).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean().values
        return hma
    
    hma_12h = calculate_hma(close_12h, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # === 4h Indicators ===
    # Donchian channels (20-period)
    def donchian_channels(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    dc_upper, dc_lower = donchian_channels(high, low, 20)
    
    # Average volume (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = 0
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or np.isnan(avg_volume[i]) or 
            np.isnan(atr[i]) or np.isnan(hma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- 12h Trend Filter ---
        hma_trend_up = close_12h[-1] > hma_12h[-1] if len(close_12h) == len(hma_12h) else hma_12h_aligned[i] > hma_12h_aligned[i-1] if i > 0 else False
        # Simplified: use slope of HMA
        if i >= 1:
            hma_slope = hma_12h_aligned[i] - hma_12h_aligned[i-1]
            hma_trend_up = hma_slope > 0
        else:
            hma_trend_up = True  # default warmup
        
        # --- Volume Confirmation ---
        volume_spike = volume[i] > 1.5 * avg_volume[i]
        
        # --- Donchian Breakout Signals ---
        breakout_up = close[i] > dc_upper[i-1]  # Close above previous upper band
        breakdown_down = close[i] < dc_lower[i-1]  # Close below previous lower band
        
        # --- Position Management (Exit Logic) ---
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                if close[i] < entry_price - 2.5 * entry_atr:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                if close[i] > entry_price + 2.5 * entry_atr:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout + Uptrend + Volume spike
        if breakout_up and hma_trend_up and volume_spike:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_atr = atr[i]
            signals[i] = SIZE
        # Short: Donchian breakdown + Downtrend + Volume spike
        elif breakdown_down and not hma_trend_up and volume_spike:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_atr = atr[i]
            signals[i] = -SIZE
    
    return signals