#!/usr/bin/env python3
"""
Experiment #264: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation

HYPOTHESIS: Daily Donchian channel breakouts capture significant price movements aligned with the weekly trend (via HMA-21). Volume confirmation ensures institutional participation, reducing false breakouts. The strategy avoids choppy regimes by requiring weekly ADX > 20. Targets 7-25 trades/year on 1d timeframe (30-100 total over 4 years) to minimize fee drag while capturing medium-term trends in both bull and bear markets. Uses discrete position sizing (0.25) and ATR-based stoploss (2*ATR(14)).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_20_1w_hma_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend and ADX regime (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HMA(21) on 1w close
    if len(df_1w) >= 21:
        close_1w = df_1w['close'].values
        hma_21_1w = calculate_hma(close_1w, 21)
        hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    else:
        hma_21_1w_aligned = np.full(n, np.nan)
    
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
        up_move = np.zeros(len(high_1w))
        down_move = np.zeros(len(low_1w))
        up_move[0] = 0
        down_move[0] = 0
        for i in range(1, len(high_1w)):
            up_move[i] = max(high_1w[i] - high_1w[i-1], 0)
            down_move[i] = max(low_1w[i-1] - low_1w[i], 0)
        
        # Smoothed DM and TR
        period = 14
        tr_sum = pd.Series(tr_1w).ewm(span=period, min_periods=period, adjust=False).mean().values
        up_sum = pd.Series(up_move).ewm(span=period, min_periods=period, adjust=False).mean().values
        down_sum = pd.Series(down_move).ewm(span=period, min_periods=period, adjust=False).mean().values
        
        # Directional Indicators
        plus_di = 100 * up_sum / tr_sum
        minus_di = 100 * down_sum / tr_sum
        
        # DX and ADX
        dx = np.zeros(len(close_1w))
        dx_sum = np.zeros(len(close_1w))
        for i in range(len(close_1w)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx_1w = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
        adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    else:
        adx_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators ===
    # Donchian Channel (20)
    def donchian_channel(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    upper_20, lower_20 = donchian_channel(high, low, 20)
    
    # ATR(14) for stoploss
    def atr(high, low, close, period):
        tr = np.zeros(len(close))
        tr[0] = high[0] - low[0]
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr_vals = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        return atr_vals
    
    atr_14 = atr(high, low, close, 14)
    
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
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or 
            np.isnan(hma_21_1w_aligned[i]) or np.isnan(adx_1w_aligned[i]) or 
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Avoid ranging markets (ADX < 20) ---
        if adx_1w_aligned[i] < 20:
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Above average volume ---
        avg_volume = np.mean(volume[max(0, i-20):i+1])
        volume_confirmed = volume[i] > avg_volume * 0.8  # At least 80% of average
        
        # --- Donchian Breakout Signals ---
        breakout_up = close[i] > upper_20[i]  # New 20-day high
        breakout_down = close[i] < lower_20[i]  # New 20-day low
        
        # --- Trend Alignment: Price vs Weekly HMA ---
        price_above_hma = close[i] > hma_21_1w_aligned[i]
        price_below_hma = close[i] < hma_21_1w_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on opposite Donchian breakout
                if close[i] < lower_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on opposite Donchian breakout
                if close[i] > upper_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Breakout above upper Donchian with volume confirmation and price above weekly HMA
        if breakout_up and volume_confirmed and price_above_hma:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        # Short: Breakout below lower Donchian with volume confirmation and price below weekly HMA
        elif breakout_down and volume_confirmed and price_below_hma:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

def calculate_hma(data, period):
    """Calculate Hull Moving Average"""
    if len(data) < period:
        return np.full_like(data, np.nan)
    
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = pd.Series(data).ewm(span=half_period, min_periods=half_period, adjust=False).mean().values
    # WMA of full period
    wma_full = pd.Series(data).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final HMA: WMA of raw_hma with sqrt_period
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma