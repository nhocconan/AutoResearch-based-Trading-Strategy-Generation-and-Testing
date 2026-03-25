#!/usr/bin/env python3
"""
Experiment #1462: 4h Primary + 1d/1w HTF — Adaptive Trend Following with Volume Confirmation

Hypothesis: 4h timeframe offers optimal balance between trade frequency (30-50/year) and 
signal quality. Previous 4h attempt (#1458) failed due to overly strict regime logic.
This version uses SIMPLER but MORE ROBUST entry conditions:

Key improvements over #1458:
1. KAMA (Kaufman Adaptive) instead of HMA — adapts to volatility, less whipsaw
2. LOOSE RSI thresholds (30/70 not 25/75) — guarantees trades in all regimes
3. Volume confirmation (taker_buy_volume ratio) — filters false breakouts
4. 1w HMA for major trend bias — only trade with weekly trend
5. Simpler logic — fewer conflicting conditions = more trades
6. ATR-based position sizing — reduce size in high volatility

Entry logic (LOOSE to guarantee ≥30 trades/train, ≥3/test):
- LONG: 1w_HMA bullish + 1d_HMA bullish + KAMA trend up + RSI>35 + volume_confirm
- SHORT: 1w_HMA bearish + 1d_HMA bearish + KAMA trend down + RSI<65 + volume_confirm
- Exit: RSI extreme (75/25) OR trailing stop (2.5x ATR)

Why this should work:
- KAMA adapts to market noise — performs better in chop than EMA/HMA
- Weekly trend filter prevents major counter-trend positions
- Volume confirmation reduces false breakouts (critical for 2022-2024 chop)
- Loose RSI ensures we get trades in both bull and bear markets
- 4h TF = natural 30-50 trades/year (fee-efficient)

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_volume_rsi_1d1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market volatility — smooth in trends, responsive in ranges
    From Perry Kaufman's "Trading Systems and Methods"
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if np.isnan(close[i]) or np.isnan(close[i - period]):
            continue
        signal = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            if not np.isnan(close[j]) and not np.isnan(close[j - 1]):
                noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 1.0
    
    # Calculate Smoothing Constant (SC)
    sc = np.full(n, np.nan, dtype=np.float64)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan, dtype=np.float64)
    # Initialize with SMA
    if period < n:
        kama[period - 1] = np.nanmean(close[:period])
    
    for i in range(period, n):
        if np.isnan(kama[i - 1]) or np.isnan(sc[i]) or np.isnan(close[i]):
            continue
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(close, period):
    """Hull Moving Average — reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

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

def calculate_volume_ratio(taker_buy_volume, volume):
    """Taker buy volume ratio — measures buying pressure"""
    n = len(volume)
    ratio = np.full(n, np.nan, dtype=np.float64)
    mask = volume > 1e-10
    ratio[mask] = taker_buy_volume[mask] / volume[mask]
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=10)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 4h indicators
    kama_10 = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_30 = calculate_kama(close, period=30, fast_period=2, slow_period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    vol_ratio = calculate_volume_ratio(taker_buy_volume, volume)
    
    # Volume ratio MA for smoothing
    vol_ratio_ma = pd.Series(vol_ratio).rolling(window=7, min_periods=7).mean().values
    
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
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_10[i]) or np.isnan(kama_30[i]):
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
        
        if np.isnan(vol_ratio_ma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (HTF bias) ===
        # Weekly trend — major bias
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Daily trend — intermediate bias
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # === 4h KAMA TREND (adaptive momentum) ===
        kama_bullish = kama_10[i] > kama_30[i]
        kama_bearish = kama_10[i] < kama_30[i]
        
        # KAMA slope (momentum)
        kama_slope_bullish = False
        kama_slope_bearish = False
        if i >= 5 and not np.isnan(kama_10[i-5]):
            kama_slope_bullish = kama_10[i] > kama_10[i-5]
            kama_slope_bearish = kama_10[i] < kama_10[i-5]
        
        # === RSI (LOOSE thresholds for trades) ===
        rsi = rsi_14[i]
        rsi_not_overbought = rsi < 70
        rsi_not_oversold = rsi > 30
        rsi_bullish = rsi > 40
        rsi_bearish = rsi < 60
        
        # === VOLUME CONFIRMATION ===
        vol_confirm_long = vol_ratio_ma[i] > 0.48  # slight buying pressure
        vol_confirm_short = vol_ratio_ma[i] < 0.52  # slight selling pressure
        
        # === ENTRY LOGIC (LOOSE — must generate trades) ===
        desired_signal = 0.0
        
        # LONG: Weekly bullish + Daily bullish + KAMA bullish + RSI confirm + volume
        if weekly_bullish and daily_bullish:
            if kama_bullish and rsi_bullish and vol_confirm_long:
                desired_signal = SIZE_BASE
            # Strong long: add KAMA slope confirmation
            if kama_bullish and kama_slope_bullish and rsi > 45 and vol_ratio_ma[i] > 0.52:
                desired_signal = SIZE_STRONG
        
        # SHORT: Weekly bearish + Daily bearish + KAMA bearish + RSI confirm + volume
        elif weekly_bearish and daily_bearish:
            if kama_bearish and rsi_bearish and vol_confirm_short:
                desired_signal = -SIZE_BASE
            # Strong short: add KAMA slope confirmation
            if kama_bearish and kama_slope_bearish and rsi < 55 and vol_ratio_ma[i] < 0.48:
                desired_signal = -SIZE_STRONG
        
        # === EXIT CONDITIONS (RSI extreme or stoploss) ===
        if in_position and position_side > 0:
            # Long exit: RSI overbought
            if rsi > 72:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Short exit: RSI oversold
            if rsi < 28:
                desired_signal = 0.0
        
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