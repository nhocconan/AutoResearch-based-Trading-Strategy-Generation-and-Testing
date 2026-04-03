#!/usr/bin/env python3
"""
Experiment #249: 4h Donchian Breakout + HMA Trend + Volume Spike + ATR Stoploss

HYPOTHESIS: Combining Donchian(20) breakouts with 4h HMA trend alignment and volume confirmation creates a robust breakout strategy that works in both bull and bear markets. The 4h HMA provides smooth trend direction, Donchian channels identify structural breakouts, volume confirms institutional participation, and ATR-based stoploss manages risk. Targets 25-50 trades/year on 4h timeframe (100-200 total over 4 years) to minimize fee drag while capturing high-probability trend continuations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close for trend filter
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for regime filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate ADX(14) on 1w data for regime filter
    if len(df_1w) >= 14:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # True Range
        tr_1w = np.zeros(len(close_1w))
        tr_1w[0] = high_1w[0] - low_1w[0]
        for i in range(1, len(close_1w)):
            tr_1w[i] = max(high_1w[i] - low_1w[i], abs(high_1w[i] - close_1w[i-1]), abs(low_1w[i] - close_1w[i-1]))
        
        # Directional Movement
        dm_plus_1w = np.zeros(len(close_1w))
        dm_minus_1w = np.zeros(len(close_1w))
        for i in range(1, len(close_1w)):
            up_move = high_1w[i] - high_1w[i-1]
            down_move = low_1w[i-1] - low_1w[i]
            dm_plus_1w[i] = up_move if up_move > down_move and up_move > 0 else 0
            dm_minus_1w[i] = down_move if down_move > up_move and down_move > 0 else 0
        
        # Smoothed values
        atr_1w = pd.Series(tr_1w).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus_1w).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus_1w).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Directional Indicators
        di_plus = np.where(atr_1w > 0, 100 * dm_plus_smooth / atr_1w, 0)
        di_minus = np.where(atr_1w > 0, 100 * dm_minus_smooth / atr_1w, 0)
        
        # DX and ADX
        dx = np.where((di_plus + di_minus) > 0, 100 * abs(di_plus - di_minus) / (di_plus + di_minus), 0)
        adx_1w = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Align to 4h timeframe
        adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    else:
        adx_1w_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # Donchian Channel(20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # HMA(21) on 4h close
    def hma(series, period):
        """Hull Moving Average"""
        if len(series) < period:
            return np.full_like(series, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = pd.Series(series).ewm(span=half_period, adjust=False).mean().values
        wma_full = pd.Series(series).ewm(span=period, adjust=False).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_values = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean().values
        return hma_values
    
    hma_21 = hma(close, 21)
    
    # Volume Spike Detector (20-period volume average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(volume_ma > 0, volume_ma, 1)
    volume_spike = volume_ratio > 2.0  # Volume at least 2x average
    
    # ATR(14) for stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_21[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in trending markets (ADX > 25) ---
        if adx_1w_aligned[i] < 25:
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price closes below HMA (trend change)
                if close[i] < hma_21[i]:
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
                # Exit if price closes above HMA (trend change)
                if close[i] > hma_21[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout above upper band + price above HMA + 1d EMA uptrend + volume spike
        if (close[i] > donchian_high[i] and 
            close[i] > hma_21[i] and 
            close[i] > ema_50_1d_aligned[i] and 
            volume_spike[i]):
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        # Short: Donchian breakout below lower band + price below HMA + 1d EMA downtrend + volume spike
        elif (close[i] < donchian_low[i] and 
              close[i] < hma_21[i] and 
              close[i] < ema_50_1d_aligned[i] and 
              volume_spike[i]):
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals