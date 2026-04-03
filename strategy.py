#!/usr/bin/env python3
"""
Experiment #247: 6h Elder Ray + 1d ADX Regime + Volume Spike

HYPOTHESIS: Combining Elder Ray (Bull/Bear Power) on 6h for momentum strength with 1d ADX regime filter (ADX>25 = trending) and volume spike confirmation creates a strategy that captures strong directional moves in both bull and bear markets. Elder Ray identifies institutional buying/selling pressure, ADX ensures we only trade in trending regimes where momentum works, and volume spike confirms participation. Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag while avoiding choppy markets where momentum fails.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ADX regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX(14) on 1d data
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr_1d = np.zeros(len(close_1d))
        tr_1d[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(close_1d)):
            tr_1d[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        
        # Directional Movement
        up_move = np.zeros(len(high_1d))
        down_move = np.zeros(len(low_1d))
        up_move[0] = 0
        down_move[0] = 0
        for i in range(1, len(high_1d)):
            up_move[i] = max(high_1d[i] - high_1d[i-1], 0)
            down_move[i] = max(low_1d[i-1] - low_1d[i], 0)
        
        # +DM and -DM
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/14)
        def wilders_smoothing(series, period):
            result = np.full_like(series, np.nan)
            if len(series) >= period:
                result[period-1] = np.mean(series[:period])
                for i in range(period, len(series)):
                    result[i] = (result[i-1] * (period-1) + series[i]) / period
            return result
        
        atr_1d = wilders_smoothing(tr_1d, 14)
        plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
        minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
        
        # DX and ADX
        dx_1d = np.zeros(len(close_1d))
        for i in range(len(close_1d)):
            if not np.isnan(plus_di_1d[i]) and not np.isnan(minus_di_1d[i]):
                denom = plus_di_1d[i] + minus_di_1d[i]
                if denom != 0:
                    dx_1d[i] = 100 * abs(plus_di_1d[i] - minus_di_1d[i]) / denom
        
        adx_1d = wilders_smoothing(dx_1d, 14)
        
        # Align to 6h timeframe
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    else:
        adx_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for EMA200 trend filter (Call ONCE before loop) ===
    if len(df_1d) >= 200:
        close_1d = df_1d['close'].values
        ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
        ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    else:
        ema_200_1d_aligned = np.full(n, np.nan)
    
    # === 6h Indicators ===
    # EMA13 and EMA21 for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA21
    bull_power = high - ema_13
    bear_power = low - ema_21
    
    # Volume spike: volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    
    warmup = 200  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade when ADX > 25 (trending market) ---
        if adx_1d_aligned[i] <= 25:
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Align with 1d EMA200 ---
        price_above_ema200 = close[i] > ema_200_1d_aligned[i]
        price_below_ema200 = close[i] < ema_200_1d_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = close[entry_bar] - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Bull Power < 0 (momentum weakening)
                if bull_power[i] < 0:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = close[entry_bar] + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Bear Power > 0 (momentum weakening)
                if bear_power[i] > 0:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Strong Bull Power + price above EMA200 + volume spike
        if bull_power[i] > 0 and price_above_ema200 and volume_spike[i]:
            in_position = True
            position_side = 1
            entry_bar = i
            signals[i] = SIZE
        # Short: Strong Bear Power + price below EMA200 + volume spike
        elif bear_power[i] < 0 and price_below_ema200 and volume_spike[i]:
            in_position = True
            position_side = -1
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals