#!/usr/bin/env python3
"""
Experiment #220: 6h Primary + 1d/1w HTF — KAMA Adaptive Trend + ATR Volatility Breakout

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). This strategy uses:
- KAMA (Kaufman Adaptive Moving Average) for trend - adapts to market efficiency, less whipsaw than HMA/EMA
- ATR volatility expansion for entry timing - enters when vol expands (trend starting)
- 1d HMA(50) for major trend bias - only trade in HTF direction
- RSI(14) momentum filter - simple threshold (not extreme) to ensure trades generate
- Weekly pivot proximity - adds confluence without over-filtering

Key differences from failed 6h attempts:
- FEWER filters (previous had 5+ confluence requirements = 0 trades)
- RSI threshold reachable (35/65 not 15/85)
- ATR expansion ratio > 1.3 (not > 2.0 which rarely triggers)
- KAMA adapts to regime automatically (no separate chop detection needed)

Position sizing: 0.25 base, 0.30 strong signals
Stoploss: 2.5x ATR trailing
Target: 30-60 trades/year, Sharpe > 0.399 (beat current 6h best)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_atr_volbreakout_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency (trend vs noise)
    More responsive in trends, smoother in chop
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    # Efficiency Ratio (ER)
    er = np.zeros(n)
    er[:] = np.nan
    
    for i in range(period, n):
        price_change = abs(close[i] - close[i-period])
        volatility = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if volatility > 1e-10:
            er[i] = price_change / volatility
        else:
            er[i] = 0.0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Initialize with SMA
    kama[period] = np.mean(close[:period+1])
    
    for i in range(period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    return kama

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_atr_ratio(atr, short_period=7, long_period=21):
    """ATR ratio - measures volatility expansion"""
    n = len(atr)
    if n < long_period:
        return np.full(n, np.nan)
    
    atr_short = pd.Series(atr).ewm(span=short_period, min_periods=short_period, adjust=False).mean().values
    atr_long = pd.Series(atr).ewm(span=long_period, min_periods=long_period, adjust=False).mean().values
    
    ratio = np.zeros(n)
    ratio[:] = np.nan
    for i in range(long_period, n):
        if atr_long[i] > 1e-10:
            ratio[i] = atr_short[i] / atr_long[i]
    
    return ratio

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for weekly trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=20)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    kama = calculate_kama(close, period=14, fast_period=2, slow_period=30)
    atr = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(atr, short_period=7, long_period=21)
    rsi = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25  # 25% base position size
    SIZE_STRONG = 0.30  # 30% for strong signals
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(atr_ratio[i]) or np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d and 1w HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === KAMA TREND (adaptive) ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        kama_slope_bull = kama[i] > kama[i-3] if i >= 3 else False
        kama_slope_bear = kama[i] < kama[i-3] if i >= 3 else False
        
        # === VOLATILITY EXPANSION (ATR ratio) ===
        vol_expansion = atr_ratio[i] > 1.25  # Vol expanding = trend starting
        vol_normal = atr_ratio[i] <= 1.25
        
        # === RSI MOMENTUM (reachable thresholds) ===
        rsi_bull = rsi[i] > 45.0 and rsi[i] < 70.0  # Bullish momentum, not overbought
        rsi_bear = rsi[i] < 55.0 and rsi[i] > 30.0  # Bearish momentum, not oversold
        rsi_neutral = rsi[i] >= 40.0 and rsi[i] <= 60.0
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === SMA TREND FILTER ===
        above_sma50 = close[i] > sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma50 = close[i] < sma_50[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG SETUP: HTF bull + KAMA bull + momentum + vol expansion OR breakout
        long_conditions = (
            htf_1d_bull and  # 1d trend up
            kama_bull and  # 6h KAMA bull
            kama_slope_bull and  # KAMA sloping up
            rsi_bull  # RSI momentum
        )
        
        # Strong long: add vol expansion or breakout
        if long_conditions:
            if vol_expansion or breakout_long:
                # Add SMA filter for extra confirmation
                if above_sma50:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT SETUP: HTF bear + KAMA bear + momentum + vol expansion OR breakout
        short_conditions = (
            htf_1d_bear and  # 1d trend down
            kama_bear and  # 6h KAMA bear
            kama_slope_bear and  # KAMA sloping down
            rsi_bear  # RSI momentum
        )
        
        # Strong short: add vol expansion or breakout
        if short_conditions:
            if vol_expansion or breakout_short:
                # Add SMA filter for extra confirmation
                if below_sma50:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # Weekly confluence boost
        if htf_1w_bull and desired_signal > 0:
            desired_signal = min(desired_signal * 1.1, SIZE_STRONG)
        elif htf_1w_bear and desired_signal < 0:
            desired_signal = max(desired_signal * 1.1, -SIZE_STRONG)
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals