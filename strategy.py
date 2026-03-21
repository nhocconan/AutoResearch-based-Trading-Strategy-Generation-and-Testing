#!/usr/bin/env python3
"""
EXPERIMENT #029 - KAMA Donchian ADX Triple HTF (12h primary, 1d/1w HTF)
================================================================================
Hypothesis: KAMA adapts to market volatility better than EMA/HMA, reducing
whipsaws in choppy markets. Donchian breakout confirmation ensures we enter
only on actual price breaks, not just indicator crossovers. ADX filter ensures
we trade only in strong trending markets (ADX > 25). Triple HTF alignment
(12h KAMA + 1d HMA + 1w HMA) provides strong trend confirmation across timeframes.

Key features:
- Primary TF: 12h (as required for this experiment)
- HTF filters: 1d HMA(50) + 1w HMA(50) for major trend alignment
- Trend: KAMA(14) adaptive moving average on 12h
- Entry: Donchian(20) breakout in trend direction
- Filter: ADX(14) > 25 (strong trend only)
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25 discrete levels with take-profit reduction
- Take profit: Reduce to half at 2R profit, trail stop at 1R

Why this differs from failed attempts:
- KAMA adapts to volatility (unlike fixed EMA/HMA)
- Donchian breakout = actual price action confirmation
- ADX filter avoids choppy markets (major cause of DD in previous strategies)
- Triple HTF alignment (12h+1d+1w) = stronger trend confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_donchian_adx_triplehtf_12h_1d_1w_v1"
timeframe = "12h"
leverage = 1.0


def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market volatility - moves fast in trends, slow in chop
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio (ER)
    change = close_s.diff(period)
    volatility = close_s.diff().abs().rolling(window=period, min_periods=period).sum()
    er = change.abs() / (volatility + 1e-10)
    er = er.fillna(0)
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i - 1]
        else:
            kama[i] = kama[i - 1] + sc.iloc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


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


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i - 1]
        minus_move = low[i - 1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Calculate TR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    
    # Smooth with Wilder's method (EMA with span=period)
    tr_s = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean()
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean()
    
    # Calculate +DI and -DI
    plus_di = 100 * plus_dm_s / (tr_s + 1e-10)
    minus_di = 100 * minus_dm_s / (tr_s + 1e-10)
    
    # Calculate DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, adjust=False, min_periods=period).mean()
    
    return adx.values


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high and lowest low over period)"""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = high[i - period + 1:i + 1].max()
        lower[i] = low[i - period + 1:i + 1].min()
    
    upper[:period - 1] = np.nan
    lower[:period - 1] = np.nan
    
    return upper, lower


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    hma_1w = calculate_hma(df_1w['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    kama = calculate_kama(close, period=14)
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.25  # Base position size (25% of capital - conservative)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    atr_at_entry = 0.0
    
    min_period = 150  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]) or
            np.isnan(kama[i]) or np.isnan(atr[i]) or np.isnan(adx[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Triple HTF trend alignment
        daily_trend = 1 if close[i] > hma_1d_aligned[i] else -1
        weekly_trend = 1 if close[i] > hma_1w_aligned[i] else -1
        kama_trend = 1 if close[i] > kama[i] else -1
        
        # All three timeframes must align for strong trend
        triple_aligned_long = (daily_trend == 1 and weekly_trend == 1 and kama_trend == 1)
        triple_aligned_short = (daily_trend == -1 and weekly_trend == -1 and kama_trend == -1)
        
        # ADX filter - only trade in strong trends (ADX > 25)
        strong_trend = adx[i] > 25
        
        # Donchian breakout confirmation
        donchian_breakout_long = close[i] > donchian_upper[i - 1] if not np.isnan(donchian_upper[i - 1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i - 1] if not np.isnan(donchian_lower[i - 1]) else False
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Triple HTF aligned + ADX strong + Donchian breakout
        if triple_aligned_long and strong_trend and donchian_breakout_long:
            target_signal = SIZE
        
        # Short entry: Triple HTF aligned + ADX strong + Donchian breakout
        elif triple_aligned_short and strong_trend and donchian_breakout_short:
            target_signal = -SIZE
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            r_value = 2.5 * atr_at_entry  # R = 2.5*ATR at entry
            
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 2 * r_value:
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 2 * r_value:
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            atr_at_entry = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                atr_at_entry = atr[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                if position_side == 1 and kama_trend == -1:
                    # KAMA trend reversed, exit long
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    atr_at_entry = 0.0
                    profit_target_hit = False
                elif position_side == -1 and kama_trend == 1:
                    # KAMA trend reversed, exit short
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    atr_at_entry = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals