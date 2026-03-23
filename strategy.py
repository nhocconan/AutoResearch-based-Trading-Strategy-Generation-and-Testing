#!/usr/bin/env python3
"""
Experiment #1227: 1d Primary + 1w HTF — KAMA Adaptive Trend + ADX + Donchian

Hypothesis: Daily timeframe needs simpler logic than lower TFs. KAMA (Kaufman Adaptive
Moving Average) automatically adjusts to market efficiency - fast in trends, slow in chop.
Combined with ADX for trend strength filter and 1w HMA for macro direction.

Key design:
(1) KAMA(10,2,30) - adapts speed based on market noise
(2) ADX(14) > 20 - only trade when trend has strength
(3) 1w HMA(21) - macro trend filter (aligns with weekly direction)
(4) Donchian(20) breakout - clean entry signal
(5) ATR(14) 3x trailing stop - risk management

Why this should work on 1d:
- Fewer false signals than lower TFs
- KAMA handles both trending and ranging without regime switches
- ADX filter prevents entries in weak trends (major failure mode)
- 1w HMA ensures we trade with macro trend
- Target: 25-40 trades/year, Sharpe > 0.6

Position sizing: 0.30 (30% of capital), discrete levels only
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_adx_donchian_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency ratio.
    Fast in trends, slow in choppy markets.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < slow_period + er_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if np.isnan(er[i]):
            continue
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    Measures trend strength (not direction). ADX > 20 = trending.
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
        else:
            minus_dm[i] = 0
    
    # Smooth with Wilder's method (EMA-like)
    tr_smooth = np.zeros(n)
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)
    
    tr_smooth[period-1] = np.sum(tr[:period])
    plus_dm_smooth[period-1] = np.sum(plus_dm[:period])
    minus_dm_smooth[period-1] = np.sum(minus_dm[:period])
    
    for i in range(period, n):
        tr_smooth[i] = tr_smooth[i-1] - tr_smooth[i-1]/period + tr[i]
        plus_dm_smooth[i] = plus_dm_smooth[i-1] - plus_dm_smooth[i-1]/period + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i-1] - minus_dm_smooth[i-1]/period + minus_dm[i]
    
    # Calculate DI+ and DI-
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    
    mask = tr_smooth > 1e-10
    di_plus[mask] = 100.0 * plus_dm_smooth[mask] / tr_smooth[mask]
    di_minus[mask] = 100.0 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period * 2 - 1, n):
        di_sum = di_plus[i] + di_minus[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(di_plus[i] - di_minus[i]) / di_sum
    
    # Smooth DX to get ADX
    adx[period * 2 - 1] = np.mean(dx[period:period*2])
    for i in range(period * 2, n):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

def calculate_donchian(high, low, period=20):
    """Donchian Channel — highest high and lowest low over period."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i - period + 1:i + 1])
        lower[i] = np.nanmin(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Hull Moving Average — reduces lag while maintaining smoothness."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    adx = calculate_adx(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # 30% of capital
    
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
            continue
        if np.isnan(kama[i]) or np.isnan(adx[i]):
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(donchian_upper[i]):
            continue
        
        # === MACRO TREND (1w HMA) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === TREND STRENGTH (ADX) ===
        trend_strong = adx[i] > 20.0  # ADX > 20 = trending market
        
        # === KAMA DIRECTION ===
        kama_bull = False
        kama_bear = False
        if not np.isnan(kama[i]) and not np.isnan(kama[i-3]):
            kama_slope = kama[i] - kama[i-3]
            kama_bull = kama_slope > 0
            kama_bear = kama_slope < 0
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = False
        donchian_breakout_down = False
        if not np.isnan(donchian_upper[i-1]) and not np.isnan(donchian_lower[i-1]):
            donchian_breakout_up = close[i] > donchian_upper[i-1]
            donchian_breakout_down = close[i] < donchian_lower[i-1]
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # LONG: Macro bull + ADX strong + KAMA bull + Donchian breakout
        if macro_bull and trend_strong and kama_bull and donchian_breakout_up:
            desired_signal = BASE_SIZE
        # LONG alternative: Macro bull + ADX strong + price > KAMA (pullback entry)
        elif macro_bull and trend_strong and kama_bull and close[i] > kama[i]:
            # Only if we had a recent pullback (price was below KAMA 2-5 bars ago)
            was_below = False
            for j in range(2, 6):
                if i - j >= 0 and not np.isnan(kama[i-j]):
                    if close[i-j] < kama[i-j]:
                        was_below = True
                        break
            if was_below:
                desired_signal = BASE_SIZE
        
        # SHORT: Macro bear + ADX strong + KAMA bear + Donchian breakout
        if macro_bear and trend_strong and kama_bear and donchian_breakout_down:
            desired_signal = -BASE_SIZE
        # SHORT alternative: Macro bear + ADX strong + price < KAMA (pullback entry)
        elif macro_bear and trend_strong and kama_bear and close[i] < kama[i]:
            # Only if we had a recent pullback (price was above KAMA 2-5 bars ago)
            was_above = False
            for j in range(2, 6):
                if i - j >= 0 and not np.isnan(kama[i-j]):
                    if close[i-j] > kama[i-j]:
                        was_above = True
                        break
            if was_above:
                desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 3x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals