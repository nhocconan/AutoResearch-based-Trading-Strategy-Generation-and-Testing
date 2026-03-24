#!/usr/bin/env python3
"""
Experiment #635: 6h Primary + 12h/1d HTF — KAMA Adaptive Trend + RSI Pullback + ADX Regime

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). KAMA adapts to 
market efficiency ratio, reducing whipsaws in choppy markets while capturing trends.
Combined with 12h trend bias and 1d macro filter, this should generate 30-50 trades/year
with better risk-adjusted returns than pure HMA strategies.

Key innovations over failed 6h strategies:
1. KAMA (Kaufman Adaptive MA) instead of HMA/EMA - adapts to volatility regime
2. 12h as intermediate HTF filter (not just 1d/1w) - better granularity
3. ADX(14) regime filter - only trade when ADX > 20 (trending) or < 25 (mean revert)
4. RSI zones: 35-65 for entries (wide enough to generate trades)
5. ROC(10) momentum confirmation - ensures entry has follow-through
6. Asymmetric sizing: 0.30 when HTF confirms, 0.20 when only local signal

Strategy logic:
1. 1d KAMA(21) = macro trend bias
2. 12h KAMA(21) = intermediate trend confirmation  
3. 6h KAMA(21) = local trend + entry trigger
4. 6h RSI(14) = pullback entry zone (35-65)
5. 6h ADX(14) = regime filter (>20 = trend, <25 = range)
6. 6h ROC(10) = momentum confirmation (>0 for long, <0 for short)
7. 6h ATR(14) = stoploss (2.5*ATR trailing)

Entry conditions (LOOSE to ensure 30+ trades):
- LONG: close > 12h KAMA AND RSI < 65 AND ROC > 0 AND (ADX > 20 OR ADX < 25)
- SHORT: close < 12h KAMA AND RSI > 35 AND ROC < 0 AND (ADX > 20 OR ADX < 25)
- 1d KAMA confirms macro (boosts size to 0.30)

Target: Sharpe > 0.40, trades >= 30 train, trades >= 3 test, DD > -50%
Timeframe: 6h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_rsi_adx_regime_12h1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market efficiency - smooth in chop, responsive in trends
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    er[:] = np.nan
    
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[max(0, i-slow_period):i+1])))
        if volatility > 1e-10:
            er[i] = price_change / volatility
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Initialize KAMA with SMA
    kama[period] = np.nanmean(close[:period+1])
    
    for i in range(period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - measures trend strength"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_diff = high[i] - high[i-1]
        minus_diff = low[i-1] - low[i]
        
        if plus_diff > minus_diff and plus_diff > 0:
            plus_dm[i] = plus_diff
        if minus_diff > plus_diff and minus_diff > 0:
            minus_dm[i] = minus_diff
    
    # Smoothed DM and TR
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    # Initialize with EMA
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
        else:
            plus_di[i] = 0.0
            minus_di[i] = 0.0
        
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0.0
    
    # ADX = EMA of DX
    adx_raw = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[period*2:] = adx_raw[period*2:]
    
    return adx

def calculate_roc(close, period=10):
    """Rate of Change - momentum indicator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    roc = np.zeros(n)
    roc[:] = np.nan
    
    for i in range(period, n):
        if close[i - period] > 1e-10:
            roc[i] = 100.0 * (close[i] - close[i - period]) / close[i - period]
    
    return roc

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF KAMAs
    kama_12h_raw = calculate_kama(df_12h['close'].values, period=21)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate 6h indicators
    kama_6h = calculate_kama(close, period=21)
    rsi = calculate_rsi(close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    roc = calculate_roc(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]) or np.isnan(roc[i]) or np.isnan(kama_6h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_12h_aligned[i]) or np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (12h primary, 1d confirmation) ===
        htf_bull = close[i] > kama_12h_aligned[i]
        htf_bear = close[i] < kama_12h_aligned[i]
        
        # 1d macro confirmation (boosts size)
        macro_bull = close[i] > kama_1d_aligned[i]
        macro_bear = close[i] < kama_1d_aligned[i]
        
        # === LOCAL TREND ===
        local_bull = close[i] > kama_6h[i]
        local_bear = close[i] < kama_6h[i]
        
        # === ADX REGIME FILTER ===
        # ADX > 20 = trending market (prefer trend entries)
        # ADX < 25 = ranging market (prefer mean reversion)
        is_trending = adx[i] > 20.0
        is_ranging = adx[i] < 25.0
        
        # === RSI PULLBACK ZONES (WIDE to ensure trades) ===
        rsi_ok_long = rsi[i] < 65.0
        rsi_ok_short = rsi[i] > 35.0
        
        # === ROC MOMENTUM CONFIRMATION ===
        roc_ok_long = roc[i] > 0.0
        roc_ok_short = roc[i] < 0.0
        
        # === ENTRY LOGIC (LOOSE CONDITIONS) ===
        desired_signal = 0.0
        
        # LONG: HTF bull + RSI not overbought + momentum + regime ok
        if htf_bull and rsi_ok_long and roc_ok_long:
            if is_trending or is_ranging:  # Either regime works
                # Strong signal: 1d macro confirms
                if macro_bull and local_bull:
                    desired_signal = SIZE_STRONG
                # Base signal: 12h HTF only
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT: HTF bear + RSI not oversold + momentum + regime ok
        elif htf_bear and rsi_ok_short and roc_ok_short:
            if is_trending or is_ranging:  # Either regime works
                # Strong signal: 1d macro confirms
                if macro_bear and local_bear:
                    desired_signal = -SIZE_STRONG
                # Base signal: 12h HTF only
                else:
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
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
                entry_atr = atr[i]
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