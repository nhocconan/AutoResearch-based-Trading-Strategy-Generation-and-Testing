#!/usr/bin/env python3
"""
Experiment #051: 4h Primary + 1d HTF — KAMA Adaptive Trend + RSI Pullback

Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency better than HMA.
In ranging markets (2022, 2025), KAMA flattens and reduces whipsaws. In trends, it follows closely.
Combined with 1d HMA bias and loose RSI thresholds, this should:
1. Reduce whipsaws in 2022 crash vs simple EMA/HMA
2. Generate MORE trades than choppiness-filtered strategies (#039, #046, #047 failed with 0 trades)
3. Work on ALL symbols (BTC/ETH/SOL) not just SOL

Key differences from #044:
- KAMA instead of HMA (adapts to volatility, less whipsaw in ranges)
- RSI thresholds: 40/60 instead of 45/55 (slightly tighter but still loose)
- Add Bollinger Band mean reversion as SECONDARY entry path
- Trailing stop: 2.0x ATR (tighter than 2.5x to protect gains)
- Size: 0.25 (more conservative through 77% crash)

Entry logic (OR conditions for trade generation):
LONG: 1d_HMA_bull AND (4h_KAMA_bull OR RSI<50 OR price<BB_lower)
SHORT: 1d_HMA_bear AND (4h_KAMA_bear OR RSI>50 OR price>BB_upper)

This ensures we ALWAYS have entry opportunities when HTF trend aligns.
Target: Sharpe>0.313, trades>30/symbol train, >3/symbol test, DD>-40%
Timeframe: 4h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_bb_dual_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market efficiency ratio (ER).
    ER=1 (trending): KAMA follows price closely
    ER=0 (ranging): KAMA flattens, reduces whipsaws
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(slow_period, n):
        price_change = abs(close[i] - close[i - slow_period])
        sum_volatility = np.sum(np.abs(np.diff(close[i - slow_period:i + 1])))
        if sum_volatility > 1e-10:
            er[i] = price_change / sum_volatility
        else:
            er[i] = 0.0
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    kama = np.full(n, np.nan)
    kama[slow_period] = close[slow_period]  # Initialize
    
    for i in range(slow_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_rsi(close, period=14):
    """RSI - momentum filter with loose thresholds"""
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
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands - for mean reversion entries"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower

def calculate_sma(close, period=200):
    """Simple Moving Average - for long-term trend filter"""
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
    
    # Calculate and align 1d HMA for HTF trend bias
    def calc_hma(close_arr, period=21):
        n = len(close_arr)
        if n < period:
            return np.full(n, np.nan)
        half = period // 2
        sqrt_p = int(np.sqrt(period))
        def wma(data, span):
            res = np.full(len(data), np.nan)
            w = np.arange(1, span + 1, dtype=np.float64)
            for i in range(span - 1, len(data)):
                res[i] = np.sum(data[i - span + 1:i + 1] * w) / np.sum(w)
            return res
        wma_half = wma(close_arr, half)
        wma_full = wma(close_arr, period)
        double_wma = 2.0 * wma_half - wma_full
        hma = wma(double_wma, sqrt_p)
        return hma
    
    hma_1d_raw = calc_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position size - conservative through crashes
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(kama_4h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h TREND (KAMA) ===
        kama_bull = close[i] > kama_4h[i]
        kama_bear = close[i] < kama_4h[i]
        
        # === RSI MOMENTUM ===
        rsi_oversold = rsi[i] < 50.0  # Loose threshold for longs
        rsi_overbought = rsi[i] > 50.0  # Loose threshold for shorts
        
        # === BOLLINGER BAND MEAN REVERSION ===
        near_bb_lower = close[i] < bb_lower[i] * 1.005 if not np.isnan(bb_lower[i]) else False
        near_bb_upper = close[i] > bb_upper[i] * 0.995 if not np.isnan(bb_upper[i]) else False
        
        # === LONG-TERM FILTER (SMA200) ===
        above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === DESIRED SIGNAL (Multiple OR conditions for trade generation) ===
        desired_signal = 0.0
        
        # LONG: 1d bull + ANY of (4h KAMA bull, RSI<50, BB lower) + SMA200 filter
        if hma_1d_bull:
            long_conditions = 0
            if kama_bull:
                long_conditions += 1
            if rsi_oversold:
                long_conditions += 1
            if near_bb_lower:
                long_conditions += 1
            
            # Need at least 1 condition + HTF bias (loose for trade generation)
            if long_conditions >= 1 and above_sma200:
                desired_signal = SIZE
        
        # SHORT: 1d bear + ANY of (4h KAMA bear, RSI>50, BB upper) + SMA200 filter
        if hma_1d_bear:
            short_conditions = 0
            if kama_bear:
                short_conditions += 1
            if rsi_overbought:
                short_conditions += 1
            if near_bb_upper:
                short_conditions += 1
            
            # Need at least 1 condition + HTF bias (loose for trade generation)
            if short_conditions >= 1 and below_sma200:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
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