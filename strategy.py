#!/usr/bin/env python3
"""
Experiment #1386: 12h Primary + 1d HTF — Asymmetric Regime Adaptive KAMA

Hypothesis: Previous 12h strategies failed because they traded symmetrically in both
directions. The 2022 crash (-77% BTC) destroyed symmetric trend followers. The key
insight from #1382 (Sharpe=0.427) was asymmetric regime behavior worked better.

Key design principles:
1. ASYMMETRIC entries: Only long in bull regime, only short in bear regime
2. KAMA adapts to volatility better than HMA/EMA (Kaufman's original design)
3. ADX confirms trend strength before entering (avoid choppy whipsaws)
4. BB Width percentile detects regime (chop vs trend)
5. RSI for entry timing only (not as primary filter)
6. ATR trailing stop 2.5x for risk management
7. Position size 0.30 = conservative for 12h volatility

Regime Logic:
- BULL: price > 1d KAMA(21) AND ADX > 25 → only LONG entries on pullbacks
- BEAR: price < 1d KAMA(21) AND ADX > 25 → only SHORT entries on retracements  
- CHOP: ADX < 20 → mean revert at BB extremes (both directions allowed)

This asymmetry should protect during crashes while capturing trends.
Target: 25-45 trades/year, Sharpe > 0.618, trades >= 30 train, >= 5 test
Timeframe: 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_adx_regime_asymmetric_1d_bb_rsi_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, fast_period=2, slow_period=30, smoothing_period=10):
    """
    Kaufman Adaptive Moving Average - adapts to market noise/volatility
    ER (Efficiency Ratio) determines smoothing constant
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < slow_period + smoothing_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan)
    for i in range(slow_period, n):
        signal = abs(close[i] - close[i - slow_period])
        noise = 0.0
        for j in range(i - slow_period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant (SC)
    sc = np.full(n, np.nan)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    for i in range(slow_period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[slow_period] = close[slow_period]
    for i in range(slow_period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]
    
    return kama

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength indicator"""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    # Calculate True Range and DM
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0.0)
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0.0)
    
    # Smooth TR and DM
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
    mask2 = (plus_di + minus_di) > 1e-10
    dx[mask2] = 100.0 * np.abs(plus_di[mask2] - minus_di[mask2]) / (plus_di[mask2] + minus_di[mask2])
    
    adx_raw = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[period*2:] = adx_raw[period*2:]
    
    return adx

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands - for mean reversion and regime detection"""
    n = len(close)
    middle = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    bandwidth = np.full(n, np.nan)
    
    if n < period:
        return middle, upper, lower, bandwidth
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        if not np.any(np.isnan(window)):
            middle[i] = np.mean(window)
            std = np.std(window, ddof=0)
            upper[i] = middle[i] + std_dev * std
            lower[i] = middle[i] - std_dev * std
            if middle[i] > 1e-10:
                bandwidth[i] = (upper[i] - lower[i]) / middle[i] * 100.0
    
    return middle, upper, lower, bandwidth

def calculate_rsi(close, period=14):
    """Relative Strength Index - for entry timing"""
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

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss sizing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bb_width_percentile(bandwidth, lookback=100):
    """Calculate percentile rank of BB Width for regime detection"""
    n = len(bandwidth)
    percentile = np.full(n, np.nan)
    
    for i in range(lookback, n):
        if not np.isnan(bandwidth[i]):
            window = bandwidth[i - lookback + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                percentile[i] = np.sum(valid <= bandwidth[i]) / len(valid) * 100.0
    
    return percentile

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF KAMA for macro trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, fast_period=2, slow_period=30, smoothing_period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (12h) indicators
    kama_12h = calculate_kama(close, fast_period=2, slow_period=30, smoothing_period=10)
    adx = calculate_adx(high, low, close, period=14)
    bb_mid, bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    bb_width_pct = calculate_bb_width_percentile(bb_width, lookback=100)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(kama_12h[i]) or np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(adx[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(bb_width_pct[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        # BULL: price above 1d KAMA (macro uptrend)
        # BEAR: price below 1d KAMA (macro downtrend)
        # CHOP: ADX < 20 (no strong trend)
        
        macro_bull = close[i] > kama_1d_aligned[i]
        macro_bear = close[i] < kama_1d_aligned[i]
        
        trend_regime = adx[i] >= 25.0
        chop_regime = adx[i] < 20.0
        
        bb_chop = bb_width_pct[i] > 70.0  # High BB width = choppy/ranging
        bb_trend = bb_width_pct[i] < 30.0  # Low BB width = trending (squeeze breakout)
        
        # === PRIMARY TREND (12h KAMA) ===
        trend_bull = close[i] > kama_12h[i]
        trend_bear = close[i] < kama_12h[i]
        
        # === RSI FOR ENTRY TIMING ===
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        rsi_neutral = 35.0 < rsi[i] < 65.0
        
        # === BB EXTREMES FOR MEAN REVERSION ===
        at_bb_lower = close[i] <= bb_lower[i] * 1.002
        at_bb_upper = close[i] >= bb_upper[i] * 0.998
        
        # === DESIRED SIGNAL - ASYMMETRIC REGIME LOGIC ===
        desired_signal = 0.0
        
        # BULL REGIME: Only LONG entries (asymmetric - no shorts in bull)
        if macro_bull and trend_regime:
            # Long on pullback to KAMA with RSI confirmation
            if trend_bull and rsi_oversold and rsi[i] > 30.0:
                desired_signal = BASE_SIZE
            # Long on BB lower touch in bull trend (dip buy)
            elif trend_bull and at_bb_lower and rsi[i] > 35.0:
                desired_signal = BASE_SIZE
            # Long breakout above KAMA with ADX confirmation
            elif close[i] > kama_12h[i] and kama_12h[i] > kama_12h[i-5] if not np.isnan(kama_12h[i-5]) else False and adx[i] > 25.0:
                desired_signal = BASE_SIZE * 0.5
        
        # BEAR REGIME: Only SHORT entries (asymmetric - no longs in bear)
        elif macro_bear and trend_regime:
            # Short on retracement to KAMA with RSI confirmation
            if trend_bear and rsi_overbought and rsi[i] < 70.0:
                desired_signal = -BASE_SIZE
            # Short on BB upper touch in bear trend (rally sell)
            elif trend_bear and at_bb_upper and rsi[i] < 65.0:
                desired_signal = -BASE_SIZE
            # Short breakdown below KAMA with ADX confirmation
            elif close[i] < kama_12h[i] and kama_12h[i] < kama_12h[i-5] if not np.isnan(kama_12h[i-5]) else False and adx[i] > 25.0:
                desired_signal = -BASE_SIZE * 0.5
        
        # CHOP REGIME: Mean reversion at BB extremes (both directions)
        elif chop_regime or bb_chop:
            # Long at BB lower with oversold RSI
            if at_bb_lower and rsi_oversold:
                desired_signal = BASE_SIZE * 0.5
            # Short at BB upper with overbought RSI
            elif at_bb_upper and rsi_overbought:
                desired_signal = -BASE_SIZE * 0.5
        
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
        if abs(desired_signal) >= BASE_SIZE * 0.4:
            if desired_signal > 0:
                final_signal = BASE_SIZE
            else:
                final_signal = -BASE_SIZE
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