#!/usr/bin/env python3
"""
Experiment #1195: 6h Primary + 12h/1d HTF — Asymmetric Regime Strategy

Hypothesis: After analyzing 985+ failures, the key insight is that BTC/ETH behave differently
in trending vs ranging markets. Simple trend-following fails in ranges (2025 bear market),
while pure mean-reversion gets destroyed in trends (2021 bull run).

This strategy uses ASYMMETRIC REGIME LOGIC:
1. 1d HMA(21) = primary trend bias (never trade against it)
2. 6h ADX(14) = regime detector (ADX>25 = trend, ADX<20 = range)
3. TREND regime: Enter pullbacks to EMA(21) in trend direction only
4. RANGE regime: Mean revert at Bollinger Band extremes (2.0 std)
5. Hysteresis: ADX must cross 25 to enter trend mode, 18 to exit (avoids whipsaw)

Key differences from failed strategies:
- NO complex regime switches (CHOP + CRSI + multiple filters = 0 trades)
- LOOSE entry conditions within each regime (guarantees trades)
- Discrete sizing (0.0, ±0.25, ±0.30) to minimize fee churn
- ATR(14) 2.5x trailing stop for risk management

Target: 30-60 trades/year, Sharpe>0.5, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_asymmetric_regime_adx_bb_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - measures trend strength"""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        plus_dm[i] = max(0, high[i] - high[i-1]) if high[i] - high[i-1] > low[i-1] - low[i] else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if low[i-1] - low[i] > high[i] - high[i-1] else 0
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    minus_di = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[:period*2] = np.nan
    return adx

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    ema_21 = calculate_ema(close, period=21)
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Regime hysteresis tracking
    in_trend_regime = False
    prev_adx = 0.0
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION with HYSTERESIS ===
        adx = adx_14[i]
        
        # Enter trend regime when ADX crosses above 25
        if adx > 25.0 and not in_trend_regime:
            in_trend_regime = True
        # Exit trend regime when ADX drops below 18 (hysteresis)
        elif adx < 18.0 and in_trend_regime:
            in_trend_regime = False
        
        prev_adx = adx
        
        # === TREND DIRECTION (Daily HMA) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 12h HMA for additional confirmation
        hma_12h_valid = not np.isnan(hma_12h_aligned[i])
        price_above_12h = hma_12h_valid and close[i] > hma_12h_aligned[i]
        price_below_12h = hma_12h_valid and close[i] < hma_12h_aligned[i]
        
        # === ASYMMETRIC ENTRY LOGIC ===
        desired_signal = 0.0
        rsi = rsi_14[i]
        
        if in_trend_regime:
            # TREND REGIME: Only enter pullbacks in trend direction
            # LONG: Price above 1d HMA + pullback to EMA21 + RSI 35-50
            if price_above_1d:
                # Pullback entry: price near EMA21 (within 1%)
                ema_dist = abs(close[i] - ema_21[i]) / ema_21[i] if ema_21[i] > 0 else 999
                if ema_dist < 0.015 and 35.0 <= rsi <= 50.0:
                    if price_above_12h:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
            
            # SHORT: Price below 1d HMA + pullback to EMA21 + RSI 50-65
            elif price_below_1d:
                ema_dist = abs(close[i] - ema_21[i]) / ema_21[i] if ema_21[i] > 0 else 999
                if ema_dist < 0.015 and 50.0 <= rsi <= 65.0:
                    if price_below_12h:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
        
        else:
            # RANGE REGIME: Mean revert at Bollinger Band extremes
            # LONG: Price at/below lower BB + RSI < 35
            if bb_lower[i] > 0:
                bb_position = (close[i] - bb_mid[i]) / (bb_upper[i] - bb_lower[i] + 1e-10)
                if close[i] <= bb_lower[i] * 1.002 and rsi < 38.0:
                    # Only long if 1d trend is neutral or bullish (don't fight strong downtrend)
                    if price_above_1d or abs(close[i] - hma_1d_aligned[i]) / hma_1d_aligned[i] < 0.02:
                        desired_signal = SIZE_BASE
            
            # SHORT: Price at/above upper BB + RSI > 62
            if bb_upper[i] > 0:
                if close[i] >= bb_upper[i] * 0.998 and rsi > 62.0:
                    # Only short if 1d trend is neutral or bearish
                    if price_below_1d or abs(close[i] - hma_1d_aligned[i]) / hma_1d_aligned[i] < 0.02:
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