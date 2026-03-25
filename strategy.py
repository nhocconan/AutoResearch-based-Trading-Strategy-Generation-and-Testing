#!/usr/bin/env python3
"""
Experiment #1408: 4h Primary + 12h/1d HTF — KAMA Adaptive Trend + ADX Momentum

Hypothesis: Previous 4h strategies failed due to:
1. HMA crossover too laggy in fast crypto moves
2. RSI pullback conditions too restrictive (generated few trades)
3. No volatility adaptation

This strategy uses:
1. KAMA(10/40) - Kaufman Adaptive Moving Average adapts to volatility
   - Fast in trends, slow in chop (perfect for crypto regime changes)
2. ADX(14) > 20 - Only trade when trend has momentum (filter whipsaws)
3. 12h KAMA direction - HTF bias to avoid counter-trend trades
4. 1d price position - Major trend filter (price above/below 1d KAMA)
5. ATR(14) 2.5x trailing stop - Protects from crash whipsaw
6. Discrete sizing: 0.0, ±0.25, ±0.30 (minimize fee churn)

Why this should beat mtf_6h_kama_trend_roc_momentum_1d_v1 (Sharpe=0.447):
- 4h TF = more entry opportunities than 6h (30-60 trades/year vs 20-40)
- KAMA adapts faster than HMA/EMA in volatile crypto
- ADX filter prevents chop losses (major killer in 2022-2023)
- LOOSE entry conditions guarantee trades (RSI > 45 / < 55, not extremes)
- 12h + 1d dual HTF filter prevents major trend violations

Entry logic (LOOSE to guarantee 30+ trades):
- LONG: 12h_KAMA bullish + 1d price > 1d_KAMA + 4h_KAMA10 > 4h_KAMA40 + ADX > 20 + RSI > 45
- SHORT: 12h_KAMA bearish + 1d price < 1d_KAMA + 4h_KAMA10 < 4h_KAMA40 + ADX > 20 + RSI < 55

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_trend_12h1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, fast_period=10, slow_period=40):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency (trend vs noise)
    """
    n = len(close)
    if n < slow_period + 1:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Efficiency Ratio (ER) = net change / sum of absolute changes
    change = np.abs(close - np.roll(close, slow_period))
    volatility = np.zeros(n)
    for i in range(slow_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-slow_period:i+1])))
    
    er = np.zeros(n)
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    er = np.clip(er, 0, 1)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[slow_period] = close[slow_period]
    
    for i in range(slow_period + 1, n):
        if not np.isnan(close[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - measures trend strength
    ADX > 20 = trending, ADX < 20 = ranging
    """
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        elif minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smoothed DM and TR (Wilder's smoothing)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    # Initialize with SMA for first period
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)
    tr_smooth = np.zeros(n)
    
    plus_dm_smooth[period-1] = np.sum(plus_dm[:period])
    minus_dm_smooth[period-1] = np.sum(minus_dm[:period])
    tr_smooth[period-1] = np.sum(tr[:period])
    
    for i in range(period, n):
        plus_dm_smooth[i] = plus_dm_smooth[i-1] - plus_dm_smooth[i-1]/period + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i-1] - minus_dm_smooth[i-1]/period + minus_dm[i]
        tr_smooth[i] = tr_smooth[i-1] - tr_smooth[i-1]/period + tr[i]
    
    # DI calculation
    mask = tr_smooth > 0
    plus_di[mask] = 100 * plus_dm_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    # DX and ADX
    di_sum = plus_di + minus_di
    mask2 = di_sum > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    # ADX = SMA of DX
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    kama_12h_raw = calculate_kama(df_12h['close'].values, fast_period=10, slow_period=40)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    kama_1d_raw = calculate_kama(df_1d['close'].values, fast_period=10, slow_period=40)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate 4h indicators
    kama_10 = calculate_kama(close, fast_period=10, slow_period=40)
    kama_40 = calculate_kama(close, fast_period=40, slow_period=80)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 150
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_10[i]) or np.isnan(kama_40[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(kama_12h_aligned[i]) or np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (12h KAMA direction) ===
        kama_12h_bullish = close[i] > kama_12h_aligned[i]
        kama_12h_bearish = close[i] < kama_12h_aligned[i]
        
        # === MAJOR TREND FILTER (1d KAMA position) ===
        price_above_1d = close[i] > kama_1d_aligned[i]
        price_below_1d = close[i] < kama_1d_aligned[i]
        
        # === 4h KAMA CROSSOVER (adaptive trend) ===
        kama_bullish = kama_10[i] > kama_40[i]
        kama_bearish = kama_10[i] < kama_40[i]
        
        # === ADX MOMENTUM FILTER ===
        adx = adx_14[i]
        adx_strong = adx > 20  # Trending market
        
        # === RSI MOMENTUM (LOOSE - guarantee trades) ===
        rsi = rsi_14[i]
        rsi_long_ok = rsi > 45  # Not too weak
        rsi_short_ok = rsi < 55  # Not too strong
        
        # === ENTRY LOGIC (LOOSE - must generate 30+ trades) ===
        desired_signal = 0.0
        
        # LONG: 12h bullish + 1d above + 4h KAMA bullish + ADX strong + RSI ok
        if kama_12h_bullish and price_above_1d and kama_bullish and adx_strong and rsi_long_ok:
            # Strong signal if ADX > 30 (very strong trend)
            if adx > 30:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT: 12h bearish + 1d below + 4h KAMA bearish + ADX strong + RSI ok
        elif kama_12h_bearish and price_below_1d and kama_bearish and adx_strong and rsi_short_ok:
            # Strong signal if ADX > 30 (very strong trend)
            if adx > 30:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
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
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals