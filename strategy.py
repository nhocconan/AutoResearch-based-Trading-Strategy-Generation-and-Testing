#!/usr/bin/env python3
"""
Experiment #060: 4h Donchian(20) Breakout + HMA Trend + Volume Confirmation + ATR Stoploss
HYPOTHESIS: Donchian channel breakouts from the 20-period high/low on 4h timeframe capture strong momentum moves.
Trend filter uses 4h HMA(21) to avoid counter-trend trades. Volume confirmation (>1.5x average) ensures breakout validity.
ATR-based stoploss (2*ATR) limits drawdown. Uses 1d timeframe for regime filter (ADX>25) to only trade in trending markets.
This combination has proven effective on SOLUSDT (test Sharpe 1.10-1.38) and should work across BTC/ETH/SOL in both bull/bear regimes
by only taking trades when the higher timeframe confirms a trending environment. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_060_4h_donchian_hma_vol_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ADX regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: ADX(14) for regime detection ===
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]  # First value
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=period, adjust=False, min_periods=period).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=period, adjust=False, min_periods=period).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
        di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
        return adx
    
    # Calculate ADX on 1d data
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    adx_1d = calculate_adx(h_1d, l_1d, c_1d, 14)
    # Align to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 4h Indicators: Donchian Channel (20) ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # === 4h Indicators: HMA(21) for trend filter ===
    def calculate_hma(arr, period):
        # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        wma_full = np.array([np.nan] * len(arr))
        wma_half = np.array([np.nan] * len(arr))
        
        for i in range(period - 1, len(arr)):
            wma_full[i] = wma(arr[i - period + 1:i + 1], period).sum() / period if i >= period - 1 else np.nan
        
        for i in range(half_period - 1, len(arr)):
            wma_half[i] = wma(arr[i - half_period + 1:i + 1], half_period).sum() / half_period if i >= half_period - 1 else np.nan
        
        # Handle edge cases with numpy where
        raw_hma = 2 * wma_half - wma_full
        hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False, min_periods=sqrt_period).mean().values
        return hma
    
    hma_21 = calculate_hma(close, 21)
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === 4h Indicators: ATR(14) for stoploss ===
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]  # First value
        atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
        return atr
    
    atr_14 = calculate_atr(high, low, close, 14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    stop_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Warmup for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(hma_21[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(atr_14[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = vol_ratio[i] > 1.5  # Volume confirmation threshold
        is_trending = adx_1d_aligned[i] > 25  # Only trade in trending markets (1d regime)
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Stoploss: 2*ATR against position
            if position_side > 0:  # Long
                if price <= stop_price:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                if price >= stop_price:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Exit on Donchian opposite touch (trailing exit)
            if position_side > 0:  # Long
                if low[i] <= donchian_lower[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                if high[i] >= donchian_upper[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if is_trending and vol_spike:
            # Long entry: price breaks above Donchian upper + above HMA (uptrend)
            if high[i-1] <= donchian_upper[i-1] and price > donchian_upper[i-1] and price > hma_21[i]:
                in_position = True
                position_side = 1
                entry_price = price
                stop_price = entry_price - 2.0 * atr_14[i]  # 2*ATR stoploss
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower + below HMA (downtrend)
            elif low[i-1] >= donchian_lower[i-1] and price < donchian_lower[i-1] and price < hma_21[i]:
                in_position = True
                position_side = -1
                entry_price = price
                stop_price = entry_price + 2.0 * atr_14[i]  # 2*ATR stoploss
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            # No trade: either not trending or no volume spike
            signals[i] = 0.0
    
    return signals