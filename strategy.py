#!/usr/bin/env python3
"""
Experiment #1487: 1d Primary + 1w HTF — KAMA Adaptive Trend with ADX Strength Filter

Hypothesis: Current best (mtf_1d_donchian_hma_rsi_1w_atr_v1, Sharpe=0.618) uses HMA trend.
KAMA (Kaufman Adaptive MA) should outperform because it adapts to volatility:
- Faster response in strong trends (high Efficiency Ratio)
- Slower/smooth in choppy markets (low Efficiency Ratio)
This reduces whipsaw in 2022 crash while capturing 2021/2023 trends.

Key components:
1. 1w HMA for macro trend bias (proven in current best)
2. 1d KAMA for adaptive trend direction (different from HMA)
3. ADX(14) > 20 for trend strength filter (not too strict)
4. RSI(14) pullback entries at 35-65 range (loose for sufficient trades)
5. ATR(14)*2.5 trailing stoploss
6. Discrete signal sizes: 0.0, ±0.25, ±0.30

Why this should beat current best:
- KAMA reduces false signals in chop vs static HMA
- ADX filter prevents entries in low-volatility traps
- Wider RSI range ensures 25-40 trades/year (not over-filtered)
- 1d timeframe = minimal fee drag (~1-2% annually)

Timeframe: 1d
HTF: 1w (call get_htf_data ONCE before loop!)
Position Size: 0.30 (discrete levels)
Target: 25-40 trades/year, Sharpe > 0.618, ALL symbols Sharpe > 0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_adx_rsi_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on Efficiency Ratio (ER)
    ER = |price change| / sum(|price changes|) over period
    High ER = trending (fast KAMA), Low ER = choppy (slow KAMA)
    """
    n = len(close)
    if n < period + slow_period + 10:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan)
    for i in range(period, n):
        if np.isnan(close[i]) or np.isnan(close[i - period]):
            continue
        signal = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            if not np.isnan(close[j]) and not np.isnan(close[j-1]):
                noise += abs(close[j] - close[j-1])
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 1.0
    
    # Calculate smoothing constant
    sc = np.full(n, np.nan)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA with SMA
    valid_start = period
    while valid_start < n and np.isnan(close[valid_start]):
        valid_start += 1
    if valid_start >= n:
        return kama
    
    kama[valid_start] = close[valid_start]
    
    # Calculate KAMA
    for i in range(valid_start + 1, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]) and not np.isnan(close[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """
    Hull Moving Average
    Reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    # WMA helper
    def wma(series, span):
        weights = np.arange(1, span + 1)
        weights = weights / weights.sum()
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i - span + 1:i + 1]).any():
                result[i] = np.sum(series[i - span + 1:i + 1] * weights)
        return result
    
    close_series = np.array(close, dtype=float)
    wma_half = wma(close_series, half)
    wma_full = wma(close_series, period)
    
    # Combine
    combined = 2.0 * wma_half - wma_full
    hma = wma(combined, sqrt_n)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index
    Measures trend strength (not direction)
    ADX > 25 = strong trend, ADX < 20 = weak/choppy
    """
    n = len(close)
    if n < period * 2 + 10:
        return np.full(n, np.nan)
    
    adx = np.full(n, np.nan)
    
    # Calculate True Range and DM
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i-1]):
            continue
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if not np.isnan(high[i-1]) and not np.isnan(low[i-1]):
            plus_dm[i] = max(0, high[i] - high[i-1]) if high[i] - high[i-1] > low[i-1] - low[i] else 0
            minus_dm[i] = max(0, low[i-1] - low[i]) if low[i-1] - low[i] > high[i] - high[i-1] else 0
    
    # Smooth with EMA
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    mask = tr_smooth > 1e-10
    plus_di[mask] = 100.0 * plus_dm_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100.0 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    # Calculate DX and ADX
    dx = np.full(n, np.nan)
    di_sum = plus_di + minus_di
    mask2 = di_sum > 1e-10
    dx[mask2] = 100.0 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    # ADX = EMA of DX
    adx_raw = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[:] = adx_raw
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i-1]):
            tr[i] = high[i] - low[i] if not np.isnan(high[i]) and not np.isnan(low[i]) else 0
        else:
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    kama_1d = calculate_kama(close, period=10)
    kama_1d_fast = calculate_kama(close, period=5)  # Faster KAMA for crossover signals
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(adx[i]) or np.isnan(kama_1d[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(kama_1d_fast[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1w HMA) - direction bias ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND (1d KAMA) ===
        kama_bull = close[i] > kama_1d[i]
        kama_bear = close[i] < kama_1d[i]
        
        # === KAMA CROSSOVER (faster signal) ===
        kama_cross_bull = kama_1d_fast[i] > kama_1d[i]
        kama_cross_bear = kama_1d_fast[i] < kama_1d[i]
        
        # === ADX TREND STRENGTH ===
        trend_strong = adx[i] > 20.0  # Loose filter for more trades
        trend_very_strong = adx[i] > 25.0
        
        # === RSI PULLBACK - Wide range for sufficient trades ===
        rsi_bullish_pullback = 35.0 <= rsi[i] <= 65.0
        rsi_bearish_pullback = 35.0 <= rsi[i] <= 65.0
        rsi_strong_bull = rsi[i] > 50.0
        rsi_strong_bear = rsi[i] < 50.0
        
        # === DESIRED SIGNAL - ADAPTIVE TREND FOLLOWING ===
        desired_signal = 0.0
        
        # LONG: Weekly bull + Daily KAMA bull + ADX confirms + RSI support
        if weekly_bull and kama_bull and trend_strong:
            if kama_cross_bull and rsi_strong_bull:
                desired_signal = BASE_SIZE  # Strong entry
            elif kama_bull and rsi[i] > 45.0:
                desired_signal = BASE_SIZE * 0.8  # Moderate entry
            elif kama_bull and rsi[i] > 40.0 and trend_very_strong:
                desired_signal = BASE_SIZE * 0.6  # Weaker but confirmed
        
        # SHORT: Weekly bear + Daily KAMA bear + ADX confirms + RSI support
        elif weekly_bear and kama_bear and trend_strong:
            if kama_cross_bear and rsi_strong_bear:
                desired_signal = -BASE_SIZE  # Strong entry
            elif kama_bear and rsi[i] < 55.0:
                desired_signal = -BASE_SIZE * 0.8  # Moderate entry
            elif kama_bear and rsi[i] < 60.0 and trend_very_strong:
                desired_signal = -BASE_SIZE * 0.6  # Weaker but confirmed
        
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
        if desired_signal >= BASE_SIZE * 0.9:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.6:
            final_signal = BASE_SIZE * 0.8
        elif desired_signal >= BASE_SIZE * 0.3:
            final_signal = BASE_SIZE * 0.5
        elif desired_signal <= -BASE_SIZE * 0.9:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.6:
            final_signal = -BASE_SIZE * 0.8
        elif desired_signal <= -BASE_SIZE * 0.3:
            final_signal = -BASE_SIZE * 0.5
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