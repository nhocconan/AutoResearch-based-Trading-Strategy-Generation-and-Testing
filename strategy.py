#!/usr/bin/env python3
"""
EXPERIMENT #114 - MTF Supertrend+MACD+BBW+RSI with Proper HTF Alignment (15m+1h+4h v3)
==================================================================================================
Hypothesis: Recent failures (#102-#113) all used improper MTF alignment or overly complex position management.
The current best (mtf_supertrend_macd_bbw_rsi_15m_1h_4h_v1, Sharpe=3.653) proves this combination works.

Key improvements from #040:
1. USE mtf_data helper for PROPER 4h alignment (critical - 46 strategies failed without this)
2. Simplified position management (no complex TP/trailing that caused bugs in #112-#113)
3. Conservative position sizing (0.30 max, discrete levels)
4. ATR stoploss at 2.5*ATR (wider than #040's 2.0*ATR to avoid whipsaws)
5. 15m entries + 1h momentum + 4h trend (3-TF proven in current best)
6. BBW regime filter to avoid choppy markets
7. RSI pullback entries in trend direction

Why this should beat #040:
- Proper HTF alignment using mtf_data (avoids SOLUSDT data gap issues)
- 3-timeframe confirmation reduces false signals
- Simpler exit logic = fewer bugs
- Based on current best performer's proven indicator combination
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_supertrend_macd_bbw_rsi_15m_1h_4h_v3"
timeframe = "15m"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(period, n):
        mid = (high[i] + low[i]) / 2
        upper_band[i] = mid + multiplier * atr[i]
        lower_band[i] = mid - multiplier * atr[i]
    
    supertrend[period] = lower_band[period]
    
    for i in range(period + 1, n):
        if trend_direction[i - 1] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i - 1])
            if close[i] < supertrend[i]:
                supertrend[i] = upper_band[i]
                trend_direction[i] = -1
            else:
                trend_direction[i] = 1
        else:
            supertrend[i] = min(upper_band[i], supertrend[i - 1])
            if close[i] > supertrend[i]:
                supertrend[i] = lower_band[i]
                trend_direction[i] = 1
            else:
                trend_direction[i] = -1
    
    return supertrend, trend_direction


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[:period + 1])
    avg_loss[period] = np.mean(loss[:period + 1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    
    bbw = np.zeros(n)
    for i in range(period - 1, n):
        if middle[i] > 0:
            bbw[i] = (upper[i] - lower[i]) / middle[i]
    
    return upper, middle, lower, bbw


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ============================================
    # 15m indicators (entry timeframe)
    # ============================================
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    supertrend_15m, st_dir_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # ============================================
    # 1h indicators (momentum filter) - USE mtf_data helper
    # ============================================
    df_1h = get_htf_data(prices, '1h')
    if df_1h is None or len(df_1h) < 50:
        return np.zeros(n)
    
    close_1h = df_1h['close'].values
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    
    macd_1h, macd_signal_1h, macd_hist_1h = calculate_macd(close_1h, fast=12, slow=26, signal=9)
    rsi_1h = calculate_rsi(close_1h, period=14)
    _, _, _, bbw_1h = calculate_bollinger_bands(close_1h, period=20, std_mult=2.0)
    
    # Align 1h indicators to 15m timeframe (auto shift for completed bars)
    macd_hist_aligned = align_htf_to_ltf(prices, df_1h, macd_hist_1h)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    bbw_1h_aligned = align_htf_to_ltf(prices, df_1h, bbw_1h)
    
    # ============================================
    # 4h indicators (trend filter) - USE mtf_data helper
    # ============================================
    df_4h = get_htf_data(prices, '4h')
    if df_4h is None or len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    supertrend_4h, st_dir_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
    macd_4h, macd_signal_4h, macd_hist_4h = calculate_macd(close_4h, fast=12, slow=26, signal=9)
    
    # Align 4h indicators to 15m timeframe
    st_dir_4h_aligned = align_htf_to_ltf(prices, df_4h, st_dir_4h)
    macd_hist_4h_aligned = align_htf_to_ltf(prices, df_4h, macd_hist_4h)
    
    # ============================================
    # Signal generation
    # ============================================
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # Thresholds
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    BBW_MIN = 0.015
    ATR_STOP_MULT = 2.5
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    stoploss_price = np.zeros(n)
    
    first_valid = max(100, 50 * 4, 50 * 16)  # Wait for all indicators to warm up
    
    for i in range(first_valid, n):
        # Check for NaN values
        if (np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or np.isnan(st_dir_15m[i]) or
            np.isnan(macd_hist_aligned[i]) or np.isnan(st_dir_4h_aligned[i]) or
            atr_15m[i] == 0):
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        price = close[i]
        atr = atr_15m[i]
        
        # ============================================
        # Check existing position for stoploss
        # ============================================
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1]
            prev_stop = stoploss_price[i - 1]
            
            # Stoploss check
            if prev_side == 1 and price < prev_stop:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                stoploss_price[i] = 0
                continue
            elif prev_side == -1 and price > prev_stop:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                stoploss_price[i] = 0
                continue
            
            # Hold position
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            stoploss_price[i] = stoploss_price[i - 1]
            continue
        
        # ============================================
        # Regime filter (BBW on 1h and 4h)
        # ============================================
        if bbw_1h_aligned[i] < BBW_MIN or bbw_15m[i] < BBW_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # ============================================
        # Trend confirmation (4h Supertrend + MACD)
        # ============================================
        trend_4h = st_dir_4h_aligned[i]
        macd_4h_bull = macd_hist_4h_aligned[i] > 0
        macd_4h_bear = macd_hist_4h_aligned[i] < 0
        
        # ============================================
        # Momentum confirmation (1h MACD + RSI)
        # ============================================
        macd_1h_bull = macd_hist_aligned[i] > 0
        macd_1h_bear = macd_hist_aligned[i] < 0
        rsi_1h_val = rsi_1h_aligned[i]
        
        # ============================================
        # Entry logic
        # ============================================
        # LONG: 4h uptrend + 1h bullish momentum + 15m pullback
        if (trend_4h == 1 and macd_4h_bull and macd_1h_bull and
            rsi_1h_val > 50 and
            RSI_LONG_MIN <= rsi_15m[i] <= RSI_LONG_MAX and
            st_dir_15m[i] == 1):
            
            signals[i] = SIZE_FULL
            position_side[i] = 1
            entry_price[i] = price
            stoploss_price[i] = price - ATR_STOP_MULT * atr
        
        # SHORT: 4h downtrend + 1h bearish momentum + 15m pullback
        elif (trend_4h == -1 and macd_4h_bear and macd_1h_bear and
              rsi_1h_val < 50 and
              RSI_SHORT_MIN <= rsi_15m[i] <= RSI_SHORT_MAX and
              st_dir_15m[i] == -1):
            
            signals[i] = -SIZE_FULL
            position_side[i] = -1
            entry_price[i] = price
            stoploss_price[i] = price + ATR_STOP_MULT * atr
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals