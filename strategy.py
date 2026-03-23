#!/usr/bin/env python3
"""
Experiment #992: 12h Primary + 1d/1w HTF — KAMA Adaptive Trend + Choppiness Regime + RSI

Hypothesis: After 717 failed strategies, 12h timeframe with adaptive trend following
should work better than fixed EMA/HMA. KAMA adapts to volatility (fast in trends,
slow in ranges). Combined with Choppiness Index regime filter and relaxed RSI entries.

Why 12h timeframe:
- Target 20-50 trades/year (minimal fee drag)
- Less noise than 4h/1h, clearer trend signals
- Proven in experiment history (982, 986 had positive returns)
- HTF (1d/1w) provides strong macro bias

Key improvements from failures:
1. KAMA instead of HMA/EMA — adapts ER (Efficiency Ratio) to market state
2. ADX threshold relaxed to >20 (not >25) to ensure sufficient trades
3. Choppiness Index as PRIMARY regime filter (not secondary)
4. RSI entry thresholds relaxed (30/70 not 25/75) for more triggers
5. Funding rate as tiebreaker only (not primary signal — caused 0 trades in #990/#991)
6. Discrete signal sizes: 0.0, ±0.25, ±0.30
7. MUST generate trades — loosened confluence requirements

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
Timeframe: 12h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_adx_chop_regime_1d1w_hma_rsi_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency (trend vs noise).
    ER = |net change| / sum of absolute changes over period
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    er = np.zeros(n)
    for i in range(period - 1, n):
        net_change = np.abs(close[i] - close[i - period + 1])
        sum_changes = np.sum(np.abs(np.diff(close[i - period + 1:i + 1])))
        if sum_changes > 1e-10:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0.0
    
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = np.zeros(n)
    for i in range(n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama[period - 1] = close[period - 1]
    for i in range(period, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """Average Directional Index — measures trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
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
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10) * 100
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10) * 100
    
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

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
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (12h) indicators
    kama_12h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    adx_12h = calculate_adx(high, low, close, period=14)
    rsi_12h = calculate_rsi(close, period=14)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align 1d HMA for medium-term trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro regime
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
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
        if np.isnan(kama_12h[i]) or np.isnan(adx_12h[i]) or np.isnan(rsi_12h[i]):
            continue
        if np.isnan(chop_12h[i]) or np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        # === MACRO REGIME (1w HTF HMA21) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === MEDIUM-TERM TREND (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === KAMA TREND DIRECTION (12h primary) ===
        kama_bullish = close[i] > kama_12h[i]
        kama_bearish = close[i] < kama_12h[i]
        
        # === TREND STRENGTH (ADX) ===
        strong_trend = adx_12h[i] > 20  # Relaxed from 25 to ensure trades
        weak_trend = adx_12h[i] <= 20
        
        # === REGIME DETECTION (Choppiness Index) ===
        ranging_regime = chop_12h[i] > 55
        trending_regime = chop_12h[i] < 45
        neutral_regime = 45 <= chop_12h[i] <= 55
        
        # === RSI SIGNALS (relaxed thresholds) ===
        rsi_oversold = rsi_12h[i] < 35
        rsi_overbought = rsi_12h[i] > 65
        rsi_extreme_oversold = rsi_12h[i] < 25
        rsi_extreme_overbought = rsi_12h[i] > 75
        rsi_neutral = 35 <= rsi_12h[i] <= 65
        
        desired_signal = 0.0
        
        # === TRENDING REGIME (CHOP < 45) — Trend Following ===
        if trending_regime and strong_trend:
            # Long: KAMA bullish + macro/medium trend support + RSI not overbought
            if kama_bullish and (macro_bull or trend_1d_bullish) and not rsi_overbought:
                desired_signal = BASE_SIZE
            # Short: KAMA bearish + macro/medium trend support + RSI not oversold
            elif kama_bearish and (macro_bear or trend_1d_bearish) and not rsi_oversold:
                desired_signal = -BASE_SIZE
            # Entry on RSI pullback in trend
            elif kama_bullish and rsi_oversold and (macro_bull or trend_1d_bullish):
                desired_signal = REDUCED_SIZE
            elif kama_bearish and rsi_overbought and (macro_bear or trend_1d_bearish):
                desired_signal = -REDUCED_SIZE
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion ===
        elif ranging_regime:
            # Long: RSI oversold + price below KAMA (oversold in range)
            if rsi_oversold and kama_bearish:
                desired_signal = REDUCED_SIZE
            # Short: RSI overbought + price above KAMA (overbought in range)
            elif rsi_overbought and kama_bullish:
                desired_signal = -REDUCED_SIZE
            # Extreme RSI alone (guarantees trades)
            elif rsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            elif rsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: KAMA direction + HTF confluence
            if kama_bullish and macro_bull and trend_1d_bullish:
                desired_signal = BASE_SIZE
            elif kama_bullish and (macro_bull or trend_1d_bullish):
                desired_signal = REDUCED_SIZE
            
            if kama_bearish and macro_bear and trend_1d_bearish:
                desired_signal = -BASE_SIZE
            elif kama_bearish and (macro_bear or trend_1d_bearish):
                desired_signal = -REDUCED_SIZE
            
            # RSI extreme as tiebreaker
            if desired_signal == 0.0:
                if rsi_extreme_oversold:
                    desired_signal = REDUCED_SIZE
                elif rsi_extreme_overbought:
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
                # Hold long if KAMA bullish and RSI not extreme overbought
                if kama_bullish and rsi_12h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if KAMA bearish and RSI not extreme oversold
                if kama_bearish and rsi_12h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if KAMA + HTF all reverse
            if kama_bearish and macro_bear and trend_1d_bearish:
                desired_signal = 0.0
            # Exit if RSI extreme overbought
            if rsi_extreme_overbought:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if KAMA + HTF all reverse
            if kama_bullish and macro_bull and trend_1d_bullish:
                desired_signal = 0.0
            # Exit if RSI extreme oversold
            if rsi_extreme_oversold:
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
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
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