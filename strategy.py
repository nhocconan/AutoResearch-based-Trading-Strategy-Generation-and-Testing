#!/usr/bin/env python3
"""
Experiment #875: 6h Primary + 1d/1w HTF — Keltner Squeeze + Fisher Transform + Dual HTF

Hypothesis: 6h timeframe sits in optimal zone between 4h (too noisy) and 12h (too slow).
Keltner Channel + Bollinger Band squeeze detects low-volatility compression before breakouts.
Ehlers Fisher Transform (period=9) catches reversal points with less lag than RSI.
Dual HTF filter (1d HMA + 1w HMA) provides strong trend bias without over-filtering.

Key innovations vs failed 6h attempts:
1. Keltner/Bollinger squeeze (TTM Squeeze style) instead of simple HMA crossover
2. Fisher Transform instead of RSI — better reversal detection in bear markets
3. Dual HTF: 1d HMA(21) + 1w HMA(50) — both must agree for strong signal
4. ADX(14) > 20 for trend confirmation (looser than failed ADX>25 attempts)
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.20, ±0.30 to minimize fee churn

Entry conditions (LOOSE to ensure ≥10 trades/train, ≥3/test):
- LONG: 1d HMA bull + 1w HMA bull + squeeze release up + Fisher > -1.5
- SHORT: 1d HMA bear + 1w HMA bear + squeeze release down + Fisher < +1.5
- No squeeze: use Fisher extremes only (Fisher < -1.5 long, Fisher > +1.5 short)

Target: Sharpe>0.45 (beat current best 0.424), trades>=10 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_keltner_fisher_dual_htf_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.zeros(n)
    diff[:] = np.nan
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_n)
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, sma, lower

def calculate_keltner(high, low, close, period=20, atr_mult=1.5):
    """Keltner Channels (ATR-based)"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = calculate_atr(high, low, close, period)
    upper = ema + atr_mult * atr
    lower = ema - atr_mult * atr
    return upper, ema, lower

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Transforms price into Gaussian normal distribution for clearer reversal signals
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Calculate typical price
    typical = (high + low + close) / 3.0
    
    # Normalize price to -1 to +1 range
    fisher_raw = np.zeros(n)
    fisher_raw[:] = np.nan
    
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        price_range = highest - lowest
        
        if price_range > 1e-10:
            normalized = 2.0 * (typical[i] - lowest) / price_range - 1.0
            # Clamp to avoid division issues
            normalized = max(-0.999, min(0.999, normalized))
            fisher_raw[i] = normalized
        else:
            fisher_raw[i] = 0.0
    
    # Apply Fisher Transform
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_signal = np.zeros(n)
    fisher_signal[:] = np.nan
    
    for i in range(period, n):
        if not np.isnan(fisher_raw[i]) and not np.isnan(fisher_raw[i-1]):
            # Fisher = 0.5 * ln((1+x)/(1-x))
            x = fisher_raw[i]
            if abs(x) < 0.999:
                fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
            
            x_prev = fisher_raw[i-1]
            if abs(x_prev) < 0.999:
                fisher_signal[i] = 0.5 * np.log((1.0 + x_prev) / (1.0 - x_prev))
    
    return fisher, fisher_signal

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_squeeze_signal(bb_upper, bb_lower, kc_upper, kc_lower):
    """
    TTM Squeeze Detection
    Squeeze ON: Bollinger Bands inside Keltner Channels (low volatility)
    Squeeze OFF: Bollinger Bands outside Keltner Channels (volatility expansion)
    Returns: 1 = squeeze on, 0 = squeeze off
    """
    n = len(bb_upper)
    squeeze = np.zeros(n)
    squeeze[:] = np.nan
    
    for i in range(n):
        if not np.isnan(bb_upper[i]) and not np.isnan(bb_lower[i]) and \
           not np.isnan(kc_upper[i]) and not np.isnan(kc_lower[i]):
            bb_width = bb_upper[i] - bb_lower[i]
            kc_width = kc_upper[i] - kc_lower[i]
            if bb_width < kc_width:
                squeeze[i] = 1.0  # Squeeze ON
            else:
                squeeze[i] = 0.0  # Squeeze OFF
    
    return squeeze

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    kc_upper, kc_mid, kc_lower = calculate_keltner(high, low, close, period=20, atr_mult=1.5)
    squeeze = calculate_squeeze_signal(bb_upper, bb_lower, kc_upper, kc_lower)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(kc_upper[i]) or np.isnan(fisher[i]):
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
        
        # === DUAL HTF BIAS (1d + 1w HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # Strong bias: both HTF agree
        htf_strong_bull = htf_1d_bull and htf_1w_bull
        htf_strong_bear = htf_1d_bear and htf_1w_bear
        
        # Weak bias: only 1d agrees (looser filter for more trades)
        htf_weak_bull = htf_1d_bull
        htf_weak_bear = htf_1d_bear
        
        # === SQUEEZE STATUS ===
        squeeze_on = squeeze[i] == 1.0
        squeeze_off = squeeze[i] == 0.0
        
        # Check squeeze release (was on, now off)
        squeeze_release_up = False
        squeeze_release_down = False
        if i > 0 and not np.isnan(squeeze[i-1]):
            if squeeze[i-1] == 1.0 and squeeze[i] == 0.0:
                # Squeeze just released - check direction
                if close[i] > bb_mid[i]:
                    squeeze_release_up = True
                elif close[i] < bb_mid[i]:
                    squeeze_release_down = True
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_bull = fisher[i] > -1.5  # Long signal
        fisher_bear = fisher[i] < 1.5   # Short signal
        fisher_strong_bull = fisher[i] > -0.5  # Stronger long
        fisher_strong_bear = fisher[i] < 0.5   # Stronger short
        
        # Fisher crossover
        fisher_cross_long = False
        fisher_cross_short = False
        if not np.isnan(fisher_signal[i]) and not np.isnan(fisher[i]):
            if fisher_signal[i] < -1.5 and fisher[i] >= -1.5:
                fisher_cross_long = True
            if fisher_signal[i] > 1.5 and fisher[i] <= 1.5:
                fisher_cross_short = True
        
        # === ADX TREND STRENGTH ===
        adx_trending = not np.isnan(adx_14[i]) and adx_14[i] > 20.0
        
        # === ENTRY LOGIC (LOOSE FOR TRADES) ===
        desired_signal = 0.0
        
        # LONG entries
        if htf_strong_bull or htf_weak_bull:
            # Squeeze release long (strongest signal)
            if squeeze_release_up and fisher_bull:
                desired_signal = SIZE_STRONG
            # Fisher crossover long
            elif fisher_cross_long and htf_strong_bull:
                desired_signal = SIZE_STRONG
            # Fisher extreme long (no squeeze)
            elif not squeeze_on and fisher[i] < -1.0 and htf_weak_bull:
                desired_signal = SIZE_BASE
            # ADX trending + Fisher bull
            elif adx_trending and fisher_strong_bull and htf_weak_bull:
                desired_signal = SIZE_BASE
        
        # SHORT entries
        elif htf_strong_bear or htf_weak_bear:
            # Squeeze release short (strongest signal)
            if squeeze_release_down and fisher_bear:
                desired_signal = -SIZE_STRONG
            # Fisher crossover short
            elif fisher_cross_short and htf_strong_bear:
                desired_signal = -SIZE_STRONG
            # Fisher extreme short (no squeeze)
            elif not squeeze_on and fisher[i] > 1.0 and htf_weak_bear:
                desired_signal = -SIZE_BASE
            # ADX trending + Fisher bear
            elif adx_trending and fisher_strong_bear and htf_weak_bear:
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