#!/usr/bin/env python3
"""
Experiment #664: 4h Primary + 12h/1d HTF — Volatility-Adaptive Regime with KAMA + RSI + ADX

Hypothesis: 4h timeframe with 12h/1d HTF filter provides optimal balance between signal 
quality and trade frequency (target 20-50 trades/year). Key innovation is volatility-adaptive 
position sizing and asymmetric regime detection that works across BTC/ETH/SOL.

Why this should beat Sharpe=0.612:
1. KAMA adapts to volatility changes better than EMA/HMA during 2022 crash and 2025 bear
2. ADX + Choppiness dual filter prevents trend-following in chop and mean-reversion in trends
3. RSI with dynamic thresholds (volatility-scaled) catches extremes better than fixed levels
4. 12h HMA for macro bias prevents counter-trend trades during strong moves
5. Volatility-adjusted position sizing reduces exposure during high vol (2022 crash survival)
6. Asymmetric logic: more aggressive longs in bull, more aggressive shorts in bear

Key differences from failed experiments:
- Simpler regime logic (2 regimes vs 3+) to avoid conflicting conditions
- Looser RSI thresholds (25/75 vs 20/80) to ensure adequate trade frequency
- Volatility scaling on position size (reduce size when ATR spikes)
- Hold logic maintains positions through minor pullbacks

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_adx_chop_voladapt_12h1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency (trend vs noise).
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    er = np.zeros(n)
    
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i+1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    kama[er_period] = np.mean(close[:er_period+1])
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
        rsi[period:] = 100 - (100 / (1 + rs[period:]))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(high[i] - high[i-1], 0)
        minus_dm[i] = max(low[i-1] - low[i], 0)
        
        if plus_dm[i] > minus_dm[i] and plus_dm[i] > 0:
            minus_dm[i] = 0
        elif minus_dm[i] > plus_dm[i] and minus_dm[i] > 0:
            plus_dm[i] = 0
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * plus_di / atr
        minus_di = 100 * minus_di / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    adx[period*2:] = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values[period:]
    
    return adx

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - high values = choppy, low = trending."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop_raw = 100.0 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(period)
        chop = np.clip(chop_raw, 0, 100)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Hull Moving Average for HTF trend."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_atr_ratio(atr, period_short=7, period_long=30):
    """ATR ratio for volatility spike detection."""
    atr_short = pd.Series(atr).ewm(span=period_short, min_periods=period_short, adjust=False).mean().values
    atr_long = pd.Series(atr).ewm(span=period_long, min_periods=period_long, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        atr_ratio = atr_short / (atr_long + 1e-10)
    
    return atr_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    rsi_4h = calculate_rsi(close, period=14)
    adx_4h = calculate_adx(high, low, close, period=14)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    atr_ratio_4h = calculate_atr_ratio(atr_4h, period_short=7, period_long=30)
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(kama_4h[i]) or np.isnan(rsi_4h[i]):
            continue
        if np.isnan(adx_4h[i]) or np.isnan(chop_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_ratio_4h[i]):
            continue
        
        # === VOLATILITY-ADAPTIVE POSITION SIZING ===
        # Reduce size when volatility spikes (ATR ratio > 2.0)
        vol_scalar = 1.0
        if atr_ratio_4h[i] > 2.0:
            vol_scalar = 0.5  # Half size during vol spikes
        elif atr_ratio_4h[i] > 1.5:
            vol_scalar = 0.75
        
        SIZE_LONG = BASE_SIZE * vol_scalar
        SIZE_SHORT = BASE_SIZE * vol_scalar * 0.85  # Slightly smaller shorts
        
        # === REGIME DETECTION ===
        # High chop = mean reversion, Low chop + High ADX = trend
        is_choppy = chop_4h[i] > 55.0
        is_trending = (chop_4h[i] < 45.0) and (adx_4h[i] > 25.0)
        
        # === HTF TREND BIAS ===
        htf_12h_bullish = close[i] > hma_12h_aligned[i]
        htf_12h_bearish = close[i] < hma_12h_aligned[i]
        
        htf_1d_bullish = close[i] > hma_1d_aligned[i]
        htf_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # Strong HTF bias when both 12h and 1d agree
        htf_strong_bull = htf_12h_bullish and htf_1d_bullish
        htf_strong_bear = htf_12h_bearish and htf_1d_bearish
        
        # === KAMA TREND ===
        kama_bullish = close[i] > kama_4h[i]
        kama_bearish = close[i] < kama_4h[i]
        
        # === RSI SIGNALS (dynamic thresholds) ===
        # In high vol, use wider thresholds
        if atr_ratio_4h[i] > 1.5:
            rsi_oversold = 20.0
            rsi_overbought = 80.0
        else:
            rsi_oversold = 30.0
            rsi_overbought = 70.0
        
        rsi_oversold_signal = rsi_4h[i] < rsi_oversold
        rsi_overbought_signal = rsi_4h[i] > rsi_overbought
        
        desired_signal = 0.0
        
        # === REGIME 1: CHOPPY MARKET (Mean Reversion) ===
        if is_choppy:
            # Long: RSI oversold + HTF not strongly bearish
            if rsi_oversold_signal and not htf_strong_bear:
                desired_signal = SIZE_LONG
            # Short: RSI overbought + HTF not strongly bullish
            elif rsi_overbought_signal and not htf_strong_bull:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 2: TRENDING MARKET (Trend Follow) ===
        elif is_trending:
            # Long: HTF bullish + KAMA bullish + RSI not overbought
            if htf_12h_bullish and kama_bullish and not rsi_overbought_signal:
                desired_signal = SIZE_LONG
            # Short: HTF bearish + KAMA bearish + RSI not oversold
            elif htf_12h_bearish and kama_bearish and not rsi_oversold_signal:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 3: TRANSITION/NEUTRAL ===
        else:
            # Use KAMA direction with RSI filter
            if kama_bullish and rsi_4h[i] < 60.0:
                desired_signal = SIZE_LONG
            elif kama_bearish and rsi_4h[i] > 40.0:
                desired_signal = -SIZE_SHORT
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if KAMA still bullish AND RSI not extremely overbought
                if kama_bullish and rsi_4h[i] < 80.0:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if KAMA still bearish AND RSI not extremely oversold
                if kama_bearish and rsi_4h[i] > 20.0:
                    desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_LONG
        elif desired_signal < 0:
            desired_signal = -SIZE_SHORT
        
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
                # Position flip
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