#!/usr/bin/env python3
"""
Experiment #1061: 4h Primary + 1d/1w HTF — KAMA Adaptive Trend + Choppiness Regime

Hypothesis: After 769+ failed experiments, the key issue is that static indicators (EMA, HMA)
fail in alternating market regimes. KAMA (Kaufman Adaptive Moving Average) adapts its
smoothing constant based on market efficiency ratio - fast in trends, slow in ranges.

This should outperform because:
1. KAMA automatically adjusts to volatility - no need for manual regime switching
2. 1d HMA21 provides macro bias without being too restrictive
3. Choppiness Index confirms regime but KAMA does the heavy lifting
4. Relaxed RSI thresholds (30-70) ensure sufficient trades
5. ATR trailing stop (2.5x) protects capital in 2022-style crashes

Key improvements over #1044:
- KAMA instead of HMA crossover (adapts automatically to regime)
- Simpler entry logic (fewer conflicting filters = more trades)
- 1w HMA for ultra-long-term bias (prevents counter-trend in major moves)
- Position size 0.25-0.30 discrete levels

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 20-50 trades/year)
Position Size: 0.25-0.30 discrete levels
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adaptive_chop_1d1w_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, fast_period=2, slow_period=30, er_period=10):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency ratio.
    Fast in trends, slow in ranges - perfect for crypto regime changes.
    
    ER = |Close - Close_n| / Sum(|Close_i - Close_i-1|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA = KAMA_prev + SC * (Close - KAMA_prev)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA with SMA of first er_period bars
    kama[er_period] = np.mean(close[:er_period + 1])
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market ranging vs trending
    CHOP > 61.8 = ranging market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        sum_atr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 1e-10:
            chop[i] = 100 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    
    avg_gain = gain_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = loss_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi[period:] = 100 - (100 / (1 + rs[period:]))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
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

def calculate_hma(series, period):
    """Hull Moving Average - faster and smoother than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength indicator."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
    
    plus_dm_series = pd.Series(plus_dm)
    minus_dm_series = pd.Series(minus_dm)
    tr_series = pd.Series(tr)
    
    smoothed_plus_dm = plus_dm_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    smoothed_minus_dm = minus_dm_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    smoothed_tr = tr_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.divide(100 * smoothed_plus_dm, smoothed_tr, out=np.zeros_like(smoothed_plus_dm), where=smoothed_tr != 0)
    minus_di = np.divide(100 * smoothed_minus_dm, smoothed_tr, out=np.zeros_like(smoothed_minus_dm), where=smoothed_tr != 0)
    
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    dx = np.divide(100 * di_diff, di_sum, out=np.zeros_like(di_diff), where=di_sum != 0)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA21 for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA21 for ultra-long-term bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (4h) indicators
    kama = calculate_kama(close, fast_period=2, slow_period=30, er_period=10)
    chop = calculate_choppiness_index(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(kama[i]) or np.isnan(chop[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(adx[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        # === MACRO TREND FILTERS ===
        # 1d HMA21 - intermediate trend
        macro_bull_1d = close[i] > hma_1d_aligned[i]
        macro_bear_1d = close[i] < hma_1d_aligned[i]
        
        # 1w HMA21 - ultra-long-term bias (prevents major counter-trend)
        macro_bull_1w = close[i] > hma_1w_aligned[i]
        macro_bear_1w = close[i] < hma_1w_aligned[i]
        
        # === KAMA TREND SIGNAL ===
        # KAMA slope (current vs 5 bars ago)
        kama_slope_bull = kama[i] > kama[i - 5] if i >= 5 else False
        kama_slope_bear = kama[i] < kama[i - 5] if i >= 5 else False
        
        # Price vs KAMA position
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range = chop[i] > 55.0
        is_trend = chop[i] < 45.0
        
        desired_signal = 0.0
        
        # === TREND MODE (CHOP < 45) ===
        if is_trend:
            # Long: KAMA bullish + price above KAMA + ADX confirms + 1d macro OK
            if kama_slope_bull and price_above_kama and adx[i] > 18 and macro_bull_1d:
                # Strong signal if 1w also bullish
                if macro_bull_1w:
                    desired_signal = BASE_SIZE
                else:
                    desired_signal = REDUCED_SIZE
            # Short: KAMA bearish + price below KAMA + ADX confirms + 1d macro OK
            elif kama_slope_bear and price_below_kama and adx[i] > 18 and macro_bear_1d:
                if macro_bear_1w:
                    desired_signal = -BASE_SIZE
                else:
                    desired_signal = -REDUCED_SIZE
        
        # === RANGE MODE (CHOP > 55) ===
        elif is_range:
            # Long: RSI oversold + price below KAMA (mean revert to KAMA) + 1d macro not bearish
            if rsi[i] < 35 and price_below_kama and not macro_bear_1d:
                desired_signal = REDUCED_SIZE
            # Short: RSI overbought + price above KAMA (mean revert to KAMA) + 1d macro not bullish
            elif rsi[i] > 65 and price_above_kama and not macro_bull_1d:
                desired_signal = -REDUCED_SIZE
        
        # === TRANSITION ZONE (45 <= CHOP <= 55) ===
        else:
            # Use KAMA direction only with stricter macro filter
            if kama_slope_bull and price_above_kama and macro_bull_1d and macro_bull_1w:
                desired_signal = REDUCED_SIZE
            elif kama_slope_bear and price_below_kama and macro_bear_1d and macro_bear_1w:
                desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if KAMA still bullish or price above KAMA
                if kama_slope_bull or price_above_kama:
                    desired_signal = BASE_SIZE if position_side > 0 else 0.0
                    if desired_signal == 0.0:
                        desired_signal = REDUCED_SIZE
            elif position_side < 0:
                # Hold short if KAMA still bearish or price below KAMA
                if kama_slope_bear or price_below_kama:
                    desired_signal = -BASE_SIZE if position_side < 0 else 0.0
                    if desired_signal == 0.0:
                        desired_signal = -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if KAMA reverses bearish AND price below KAMA
            if kama_slope_bear and price_below_kama:
                desired_signal = 0.0
            # Exit long if 1d macro turns bearish
            if macro_bear_1d and rsi[i] > 60:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if KAMA reverses bullish AND price above KAMA
            if kama_slope_bull and price_above_kama:
                desired_signal = 0.0
            # Exit short if 1d macro turns bullish
            if macro_bull_1d and rsi[i] < 40:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= 0.25 else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -0.25 else -REDUCED_SIZE
        
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
                # Flip position
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