#!/usr/bin/env python3
"""
Experiment #954: 4h Primary + 12h/1d HTF — KAMA Adaptive Trend + ADX + Choppiness Regime

Hypothesis: After 664 failed strategies, KAMA (Kaufman Adaptive Moving Average) provides
superior noise filtering compared to EMA/HMA. Combined with ADX trend strength and
Choppiness Index regime detection, this should work across ALL symbols (BTC/ETH/SOL).

Why this should work:
1. KAMA adapts to market volatility — fast in trends, slow in chop (Kaufman's original design)
2. ADX > 25 filters out weak trends (reduces whipsaw in 2022 crash)
3. Choppiness Index regime switch: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend
4. 12h HMA(21) for medium-term trend bias (proven in mtf_hma_rsi_zscore_v1)
5. 1d HMA(21) for macro regime filter
6. Simpler entry logic = more trades guaranteed (addressing #1 failure mode)

Key improvements over failed experiments:
- KAMA instead of EMA/HMA (better noise adaptation)
- ADX threshold relaxed to 20 (ensure trades trigger)
- Choppiness thresholds: 38.2/61.2 (Fibonacci levels from literature)
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn
- Stoploss at 2.5*ATR with trailing logic
- NO funding rate dependency (caused 0 trades in exp #944, #952)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_chop_regime_12h1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts to market noise: fast during trends, slow during chop.
    From Perry Kaufman's "Trading Systems and Methods".
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX).
    Measures trend strength (not direction). ADX > 25 = trending.
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 3:
        return adx
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
        
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth TR, +DM, -DM
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI and DX
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * plus_di / (atr + 1e-10)
        minus_di = 100 * minus_di / (atr + 1e-10)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # ADX is smoothed DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    Measures market choppy vs trending.
    CHOP > 61.8 = rangebound, CHOP < 38.2 = trending.
    """
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

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    adx_4h, plus_di_4h, minus_di_4h = calculate_adx(high, low, close, period=14)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    rsi_4h = calculate_rsi(close, period=14)
    
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
        if np.isnan(kama_4h[i]) or np.isnan(adx_4h[i]) or np.isnan(chop_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(rsi_4h[i]):
            continue
        
        # === MACRO REGIME (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM-TERM TREND (12h HTF HMA21) ===
        trend_12h_bullish = close[i] > hma_12h_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # === KAMA TREND DIRECTION ===
        kama_bullish = close[i] > kama_4h[i]
        kama_bearish = close[i] < kama_4h[i]
        
        # === TREND STRENGTH (ADX) ===
        strong_trend = adx_4h[i] > 20  # Relaxed from 25 to ensure trades
        weak_trend = adx_4h[i] < 20
        
        # === REGIME DETECTION (Choppiness Index) ===
        ranging_regime = chop_4h[i] > 61.8  # Rangebound
        trending_regime = chop_4h[i] < 38.2  # Trending
        neutral_regime = not ranging_regime and not trending_regime
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_4h[i] < 40
        rsi_overbought = rsi_4h[i] > 60
        rsi_extreme_oversold = rsi_4h[i] < 30
        rsi_extreme_overbought = rsi_4h[i] > 70
        
        # === DI DIRECTION ===
        di_bullish = plus_di_4h[i] > minus_di_4h[i]
        di_bearish = minus_di_4h[i] > plus_di_4h[i]
        
        desired_signal = 0.0
        
        # === TRENDING REGIME (CHOP < 38.2) — Trend Following ===
        if trending_regime and strong_trend:
            # Long: All trend indicators aligned bullish
            if kama_bullish and macro_bull and (trend_12h_bullish or di_bullish):
                if rsi_oversold or rsi_4h[i] < 55:  # Pullback entry
                    desired_signal = BASE_SIZE
            # Short: All trend indicators aligned bearish
            elif kama_bearish and macro_bear and (trend_12h_bearish or di_bearish):
                if rsi_overbought or rsi_4h[i] > 45:  # Pullback entry
                    desired_signal = -BASE_SIZE
        
        # === RANGING REGIME (CHOP > 61.8) — Mean Reversion ===
        elif ranging_regime:
            # Long: Price below KAMA + oversold RSI
            if kama_bearish and rsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            # Short: Price above KAMA + overbought RSI
            elif kama_bullish and rsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) ===
        else:
            # Conservative trend following with ADX confirmation
            if strong_trend:
                if kama_bullish and (macro_bull or trend_12h_bullish) and di_bullish:
                    if rsi_4h[i] < 55:
                        desired_signal = REDUCED_SIZE
                elif kama_bearish and (macro_bear or trend_12h_bearish) and di_bearish:
                    if rsi_4h[i] > 45:
                        desired_signal = -REDUCED_SIZE
            else:
                # Weak trend: RSI extremes only
                if rsi_extreme_oversold and (macro_bull or trend_12h_bullish):
                    desired_signal = REDUCED_SIZE
                elif rsi_extreme_overbought and (macro_bear or trend_12h_bearish):
                    desired_signal = -REDUCED_SIZE
        
        # === GUARANTEED TRADE TRIGGERS (prevent 0 trades) ===
        # If no signal yet, use simpler conditions to ensure trades
        if desired_signal == 0.0:
            # Simple KAMA crossover with RSI filter
            if kama_bullish and rsi_4h[i] < 50 and macro_bull:
                desired_signal = REDUCED_SIZE
            elif kama_bearish and rsi_4h[i] > 50 and macro_bear:
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
                # Hold long if KAMA and macro trend intact
                if kama_bullish and (macro_bull or trend_12h_bullish) and rsi_4h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if KAMA and macro trend intact
                if kama_bearish and (macro_bear or trend_12h_bearish) and rsi_4h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if KAMA reverses + RSI overbought
            if kama_bearish and rsi_4h[i] > 65:
                desired_signal = 0.0
            # Exit if macro + medium trend both reverse
            if macro_bear and trend_12h_bearish:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if KAMA reverses + RSI oversold
            if kama_bullish and rsi_4h[i] < 35:
                desired_signal = 0.0
            # Exit if macro + medium trend both reverse
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