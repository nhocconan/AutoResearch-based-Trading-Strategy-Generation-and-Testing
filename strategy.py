#!/usr/bin/env python3
"""
Experiment #984: 4h Primary + 12h/1d HTF — KAMA Adaptive Trend + ADX Strength + Regime Filter

Hypothesis: After 710+ failed strategies, the key is SIMPLICITY + ADAPTIVITY.
KAMA (Kaufman Adaptive Moving Average) automatically adjusts to market noise,
outperforming fixed EMA/HMA in both trending and ranging conditions.

Key insights from research:
1. KAMA trend + ADX filter worked for ETH (Sharpe +0.755) — use as base
2. ADX > 20 filters weak trends (reduces whipsaw in 2022 crash)
3. 12h HMA(21) provides medium-term bias without overfitting
4. 1d HMA(21) macro filter prevents counter-trend trades in strong regimes
5. Choppiness Index switches between trend-follow and mean-revert modes
6. RELAXED entry conditions ensure >= 30 trades/train, >= 3/test

Why this differs from failed experiments:
- SIMPLER logic: KAMA crossover + ADX + HTF bias (not 10+ conflicting filters)
- NO funding rate dependency (data alignment issues caused failures)
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn
- Hold logic maintains position through minor pullbacks
- Stoploss at 2.5x ATR protects from 2022-style crashes

Entry conditions (relaxed to ensure trades):
- Long: KAMA10 > KAMA40 + ADX > 18 + price > 12h HMA OR (range + RSI < 35)
- Short: KAMA10 < KAMA40 + ADX > 18 + price < 12h HMA OR (range + RSI > 65)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_regime_12h1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average.
    Adapts smoothing based on market noise (Efficiency Ratio).
    ER = |net change| / sum of absolute changes over period
    SC = [ER * (fast SC - slow SC) + slow SC]^2
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period - 1, n):
        net_change = np.abs(close[i] - close[i - period + 1])
        sum_changes = np.sum(np.abs(np.diff(close[i - period + 1:i + 1])))
        if sum_changes > 1e-10:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Calculate KAMA
    kama[period - 1] = close[period - 1]
    for i in range(period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index — measures trend strength.
    ADX > 25 = strong trend, ADX < 20 = weak/ranging.
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 3:
        return adx
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smoothed DM and TR (Wilder's smoothing)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    atr = np.zeros(n)
    
    # Initial values (first period)
    atr[period-1] = np.sum(tr[1:period]) / period
    plus_di[period-1] = 100 * np.sum(plus_dm[1:period]) / atr[period-1] if atr[period-1] > 1e-10 else 0
    minus_di[period-1] = 100 * np.sum(minus_dm[1:period]) / atr[period-1] if atr[period-1] > 1e-10 else 0
    
    # Wilder's smoothing for rest
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        plus_smooth = (plus_di[i-1] * (period - 1) / 100 * atr[i-1] + plus_dm[i]) / atr[i] if atr[i] > 1e-10 else 0
        minus_smooth = (minus_di[i-1] * (period - 1) / 100 * atr[i-1] + minus_dm[i]) / atr[i] if atr[i] > 1e-10 else 0
        plus_di[i] = 100 * plus_smooth if atr[i] > 1e-10 else 0
        minus_di[i] = 100 * minus_smooth if atr[i] > 1e-10 else 0
    
    # DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0
    
    # ADX = smoothed DX
    adx[period*2-1] = np.mean(dx[period:period*2])
    for i in range(period*2, n):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx, plus_di, minus_di, atr

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index — measures market choppy vs trending."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    kama_fast_4h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_slow_4h = calculate_kama(close, period=40, fast_period=2, slow_period=30)
    adx_4h, plus_di_4h, minus_di_4h, atr_4h = calculate_adx(high, low, close, period=14)
    rsi_4h = calculate_rsi(close, period=14)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    
    # Calculate and align 12h HMA for medium-term trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 1d HMA for macro regime
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
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
        if np.isnan(kama_fast_4h[i]) or np.isnan(kama_slow_4h[i]):
            continue
        if np.isnan(adx_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(rsi_4h[i]) or np.isnan(chop_4h[i]):
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # === MACRO REGIME (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM-TERM TREND (12h HTF HMA21) ===
        trend_12h_bullish = close[i] > hma_12h_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # === KAMA TREND SIGNAL ===
        kama_bullish = kama_fast_4h[i] > kama_slow_4h[i]
        kama_bearish = kama_fast_4h[i] < kama_slow_4h[i]
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_4h[i] > 18  # Relaxed from 20 to ensure trades
        adx_very_strong = adx_4h[i] > 25
        
        # === CHOPPINESS REGIME ===
        ranging_regime = chop_4h[i] > 50  # Relaxed threshold
        trending_regime = chop_4h[i] < 40
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_4h[i] < 40  # Relaxed from 35
        rsi_overbought = rsi_4h[i] > 60  # Relaxed from 65
        rsi_extreme_oversold = rsi_4h[i] < 30
        rsi_extreme_overbought = rsi_4h[i] > 70
        
        desired_signal = 0.0
        
        # === TRENDING REGIME (CHOP < 40) — Trend Following ===
        if trending_regime and adx_strong:
            # Long: KAMA bullish + ADX strong + 12h trend support
            if kama_bullish and (trend_12h_bullish or macro_bull):
                desired_signal = BASE_SIZE
            # Long: KAMA bullish + ADX very strong (override HTF)
            elif kama_bullish and adx_very_strong:
                desired_signal = REDUCED_SIZE
            
            # Short: KAMA bearish + ADX strong + 12h trend support
            if kama_bearish and (trend_12h_bearish or macro_bear):
                desired_signal = -BASE_SIZE
            # Short: KAMA bearish + ADX very strong (override HTF)
            elif kama_bearish and adx_very_strong:
                desired_signal = -REDUCED_SIZE
        
        # === RANGING REGIME (CHOP > 50) — Mean Reversion ===
        elif ranging_regime:
            # Long: RSI oversold + macro/12h support
            if rsi_oversold and (trend_12h_bullish or macro_bull):
                desired_signal = REDUCED_SIZE
            # Long: RSI extreme oversold (stronger signal)
            elif rsi_extreme_oversold:
                desired_signal = BASE_SIZE
            
            # Short: RSI overbought + macro/12h resistance
            if rsi_overbought and (trend_12h_bearish or macro_bear):
                desired_signal = -REDUCED_SIZE
            # Short: RSI extreme overbought (stronger signal)
            elif rsi_extreme_overbought:
                desired_signal = -BASE_SIZE
        
        # === NEUTRAL REGIME (40 <= CHOP <= 50) ===
        else:
            # Conservative: KAMA + ADX + HTF confluence
            if kama_bullish and adx_strong and trend_12h_bullish:
                desired_signal = REDUCED_SIZE
            elif kama_bullish and adx_very_strong:
                desired_signal = REDUCED_SIZE
            
            if kama_bearish and adx_strong and trend_12h_bearish:
                desired_signal = -REDUCED_SIZE
            elif kama_bearish and adx_very_strong:
                desired_signal = -REDUCED_SIZE
            
            # Secondary: RSI extremes
            if rsi_extreme_oversold and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            if rsi_extreme_overbought and desired_signal == 0:
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
                # Hold long if KAMA still bullish or ADX strong
                if kama_bullish or (adx_strong and trend_12h_bullish):
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if KAMA still bearish or ADX strong
                if kama_bearish or (adx_strong and trend_12h_bearish):
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if KAMA reverses + ADX weakens
            if kama_bearish and adx_4h[i] < 15:
                desired_signal = 0.0
            # Exit if macro + 12h both bearish
            if macro_bear and trend_12h_bearish:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if KAMA reverses + ADX weakens
            if kama_bullish and adx_4h[i] < 15:
                desired_signal = 0.0
            # Exit if macro + 12h both bullish
            if macro_bull and trend_12h_bullish:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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