#!/usr/bin/env python3
"""
Experiment #094: 1h Donchian(20) Breakout + 4h/1d HMA Trend Filter + Volume Spike + Session Filter + ATR Stoploss
HYPOTHESIS: 1h Donchian breakouts aligned with 4h and 1d HMA trend direction capture momentum with HTF alignment. 
Volume confirmation (>1.3x average) and session filter (08-20 UTC) reduce false signals. 
ATR-based stoploss (2.0x) manages risk. Using 4h/1d for trend direction and 1h for precise entry timing 
targets 60-150 total trades over 4 years (15-37/year) to minimize fee drag while maintaining statistical validity.
Works in both bull/bear by taking breakouts in direction of higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_094_1h_donchian_breakout_4h_1d_hma_volume_session_atr_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === Indicators: Donchian Channels (20-period) on 1h ===
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
    
    # === HTF: 4h HMA(21) for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values.astype(np.float64)
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
    
    hma_4h = calculate_hma(close_4h, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)  # shift(1) applied
    
    # === HTF: 1d HMA(21) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values.astype(np.float64)
    hma_1d = calculate_hma(close_1d, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)  # shift(1) applied
    
    # === Session filter: 08-20 UTC (pre-compute for efficiency) ===
    # prices.index is DatetimeIndex with timezone-naive UTC timestamps
    hours = prices.index.hour
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size (discrete level to minimize churn)
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Warmup for Donchian, ATR, volume, HMAs stability
    
    for i in range(warmup, n):
        # Skip if any critical values are NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = vol_ratio[i] > 1.3  # Volume spike: 1.3x average
        price_above_htf = price > hma_4h_aligned[i] and price > hma_1d_aligned[i]
        price_below_htf = price < hma_4h_aligned[i] and price < hma_1d_aligned[i]
        in_session = (8 <= hours[i] <= 20)  # UTC 08-20
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_price = entry_price - 2.0 * entry_atr
                if price < stop_price:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_price = entry_price + 2.0 * entry_atr
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
        # Requires session filter to reduce noise trades
        if in_session:
            # Long entry: price breaks above Donchian upper with volume AND price above both 4h and 1d HMA (uptrend)
            if price > donchian_upper[i-1] and vol_spike and price_above_htf:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower with volume AND price below both 4h and 1d HMA (downtrend)
            elif price < donchian_lower[i-1] and vol_spike and price_below_htf:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals