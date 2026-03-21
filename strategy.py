#!/usr/bin/env python3
"""
EXPERIMENT #028 - KAMA Donchian Breakout with Dual HTF Filter (4h primary, 1d/1w HTF)
================================================================================
Hypothesis: KAMA adapts to market volatility better than fixed EMAs, reducing whipsaws.
Donchian channel breakouts capture momentum moves on 4h timeframe. Dual HTF filter
(1d HMA50 + 1w HMA50) ensures we only trade in direction of major trends. ADX(14) > 25
filters out choppy markets. Volume confirmation ensures breakouts have participation.

Key features:
- Primary TF: 4h
- HTF filters: 1d HMA(50) + 1w HMA(50) for dual trend alignment
- Trend: KAMA(10) adaptive moving average
- Entry: Donchian(20) breakout with volume confirmation
- Filter: ADX(14) > 25 (trending market only)
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25 base, discrete levels (0.0, ±0.25, ±0.12)
- Take profit: Reduce to half at 2R profit
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_donchian_adx_dualhtf_4h_1d_1w_v1"
timeframe = "4h"
leverage = 1.0


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(1, period + 1):
            noise += abs(close[i - j + 1] - close[i - j])
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    sc = (er * (2 / (fast_period + 1) - 2 / (slow_period + 1)) + 2 / (slow_period + 1)) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    # Calculate True Range and DM
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i - 1])
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(0, low[i - 1] - low[i])
        else:
            minus_dm[i] = 0
    
    # Smooth TR and DM
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values / (atr + 1e-10) * 100
    minus_di = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values / (atr + 1e-10) * 100
    
    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx[period*2:] = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values[period*2:]
    
    return adx


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel"""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF HMAs
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    hma_1w = calculate_hma(df_1w['close'].values, 50)
    
    # Align HTF data to LTF
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    kama = calculate_kama(close, 10)
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Calculate volume SMA for confirmation
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.25  # Base position size (25% of capital)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Track position state
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(kama[i]) or np.isnan(atr[i]) or np.isnan(adx[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]) or
            np.isnan(volume_sma[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Dual HTF trend filter
        daily_trend = 1 if close[i] > hma_1d_aligned[i] else -1
        weekly_trend = 1 if close[i] > hma_1w_aligned[i] else -1
        
        # KAMA trend direction
        kama_trend = 1 if close[i] > kama[i] else -1
        
        # ADX filter (trending market only)
        adx_valid = adx[i] > 25
        
        # Volume confirmation
        volume_valid = volume[i] > volume_sma[i]
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i - 1]  # Break above previous upper
        breakout_short = close[i] < donchian_lower[i - 1]  # Break below previous lower
        
        # Determine target signal
        target_signal = 0.0
        
        # Long entry: All filters aligned bullish + breakout
        if (daily_trend == 1 and weekly_trend == 1 and kama_trend == 1 and
            adx_valid and volume_valid and breakout_long):
            target_signal = SIZE
        
        # Short entry: All filters aligned bearish + breakout
        elif (daily_trend == -1 and weekly_trend == -1 and kama_trend == -1 and
              adx_valid and volume_valid and breakout_short):
            target_signal = -SIZE
        
        # Stoploss and take profit logic
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * entry_atr:  # 2R profit
                        take_profit_triggered = True
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                if not profit_target_hit:
                    if close[i] <= entry_price - 5.0 * entry_atr:  # 2R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            if target_signal != 0.0 and position_side == 0:
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                entry_atr = atr[i]
                profit_target_hit = False
            elif position_side != 0:
                # Check trend reversal for exit
                if position_side == 1 and kama_trend == -1:
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 0.0
                    profit_target_hit = False
                elif position_side == -1 and kama_trend == 1:
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 0.0
                    profit_target_hit = False
                else:
                    signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals