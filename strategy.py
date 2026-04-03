#!/usr/bin/env python3
"""
Experiment #090: 1d Donchian(20) Breakout + 1w HMA Trend Filter + Volume Spike + ATR Stoploss
HYPOTHESIS: Donchian breakouts on 1d aligned with 1w HMA(21) trend direction capture strong momentum with better trend alignment than lower timeframes. 
Volume confirmation (>1.5x average) filters false breakouts. ATR-based stoploss (2.5x) manages risk. 
Uses 1w timeframe for trend filter to reduce noise and improve signal quality. Target: 30-100 trades over 4 years (7-25/year).
Works in both bull/bear by taking breakouts in direction of higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_090_1d_donchian_breakout_1w_hma_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === Indicators: Donchian Channels (20-period) on 1d ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # === Indicators: ATR(14) for stoploss ===
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    # === Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === HTF: 1w HMA(21) for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values.astype(np.float64)
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    def calculate_wma(data, period):
        if period <= 1:
            return data.copy()
        weights = np.arange(1, period + 1)
        return pd.Series(data).rolling(window=period, min_periods=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True).values
    
    def calculate_hma(data, period):
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = calculate_wma(data, half_period)
        wma_full = calculate_wma(data, period)
        raw_hma = 2 * wma_half - wma_full
        hma = calculate_wma(raw_hma, sqrt_period)
        return hma
    
    hma_1w = calculate_hma(close_1w, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)  # shift(1) applied
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Warmup for Donchian, ATR, volume, HMA stability
    
    for i in range(warmup, n):
        # Skip if any critical values are NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ratio[i]) or np.isnan(hma_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = vol_ratio[i] > 1.5  # Volume spike: 1.5x average
        price_above_hma = price > hma_1w_aligned[i]
        price_below_hma = price < hma_1w_aligned[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_price = entry_price - 2.5 * entry_atr
                if price < stop_price:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_price = entry_price + 2.5 * entry_atr
                if price > stop_price:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Donchian opposite break exit (with volume confirmation)
            if position_side > 0:  # Long
                if price < donchian_lower[i-1] and vol_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                if price > donchian_upper[i-1] and vol_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 1 bar to prevent whipsaw
            if bars_since_entry < 1:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Long entry: price breaks above Donchian upper with volume AND price above 1w HMA (uptrend)
        if price > donchian_upper[i-1] and vol_spike and price_above_hma:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_atr = atr[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short entry: price breaks below Donchian lower with volume AND price below 1w HMA (downtrend)
        elif price < donchian_lower[i-1] and vol_spike and price_below_hma:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_atr = atr[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals