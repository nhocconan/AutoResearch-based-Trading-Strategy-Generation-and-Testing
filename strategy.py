#!/usr/bin/env python3
"""
Experiment #1283: 6h Primary + 1d/1w HTF — Adaptive Regime Strategy (CHOP + HMA + RSI)

Hypothesis: Pure trend following fails in bear/range markets (2022 crash, 2025 bear).
This strategy ADAPTS to regime using CHOP(14) choppiness index, with different logic
for trend vs range markets. Key innovations:

1. CHOP(14) regime detection on 6h: CHOP>61.8=range (mean revert), CHOP<38.2=trend
2. 1d HMA(21) for intermediate trend bias (only trade with daily direction)
3. 1w HMA(21) for ultra-long-term bias (avoid counter-weekly trades)
4. Regime-adaptive entries:
   - TREND: HMA cross + ROC momentum confirmation
   - RANGE: RSI extremes (RSI<30 long, RSI>70 short) at support/resistance
5. ATR(14) 2.5x trailing stop for all positions
6. LOOSE thresholds to guarantee 30-60 trades/year on 6h

Why this should beat current best (Sharpe=0.447):
- Adapts to 2022 crash (range regime = mean reversion works)
- Adapts to 2025 bear (weekly bias prevents counter-trend longs)
- 6h timeframe = natural 30-60 trades/year (fee-friendly)
- Dual HTF (1d+1w) = strong bias without over-filtering

Entry logic (LOOSE to guarantee trades):
- TREND LONG: CHOP<38.2 + 1d_HMA rising + 1w_HMA bullish + 6h_HMA cross up + ROC>3
- TREND SHORT: CHOP<38.2 + 1d_HMA falling + 1w_HMA bearish + 6h_HMA cross down + ROC<-3
- RANGE LONG: CHOP>61.8 + RSI<35 + price>1d_HMA (pullback in uptrend)
- RANGE SHORT: CHOP>61.8 + RSI>65 + price<1d_HMA (rally in downtrend)

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_adaptive_regime_chop_hma_rsi_1d1w_v1"
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_chop(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR for each bar
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_roc(close, period=10):
    """Rate of Change - momentum indicator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    roc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if close[i - period] != 0:
            roc[i] = ((close[i] - close[i - period]) / close[i - period]) * 100.0
    
    return roc

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_chop(high, low, close, period=14)
    roc_10 = calculate_roc(close, period=10)
    hma_6h = calculate_hma(close, period=21)
    
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
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
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
        
        if np.isnan(hma_6h[i]) or np.isnan(roc_10[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (CHOP) ===
        chop = chop_14[i]
        is_trend_regime = chop < 38.2
        is_range_regime = chop > 61.8
        
        # === TREND DIRECTION (1d HMA slope + 1w HMA bias) ===
        # 1d HMA slope (compare to 3 bars ago for stability)
        hma_1d_slope = 0.0
        if i >= 3 and not np.isnan(hma_1d_aligned[i-3]):
            hma_1d_slope = hma_1d_aligned[i] - hma_1d_aligned[i-3]
        
        # 1w HMA bias (ultra long term)
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # 1d price vs 1d HMA
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 6h price vs 6h HMA for local confirmation
        price_above_6h = close[i] > hma_6h[i]
        price_below_6h = close[i] < hma_6h[i]
        
        # 6h HMA slope
        hma_6h_slope = 0.0
        if i >= 3 and not np.isnan(hma_6h[i-3]):
            hma_6h_slope = hma_6h[i] - hma_6h[i-3]
        
        # === MOMENTUM (ROC) ===
        roc = roc_10[i]
        rsi = rsi_14[i]
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # TREND REGIME: Follow the trend with momentum confirmation
        if is_trend_regime:
            # LONG: 1d rising + 1w bullish + 6h HMA rising + ROC positive
            if hma_1d_slope > 0 and price_above_1w and hma_6h_slope > 0:
                if roc > 2.0:  # Very loose momentum
                    if roc > 6.0:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
            
            # SHORT: 1d falling + 1w bearish + 6h HMA falling + ROC negative
            elif hma_1d_slope < 0 and price_below_1w and hma_6h_slope < 0:
                if roc < -2.0:  # Very loose momentum
                    if roc < -6.0:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
        
        # RANGE REGIME: Mean reversion at extremes
        elif is_range_regime:
            # LONG: RSI oversold + price above 1d HMA (pullback in uptrend)
            if rsi < 35 and price_above_1d:
                desired_signal = SIZE_BASE
            
            # SHORT: RSI overbought + price below 1d HMA (rally in downtrend)
            elif rsi > 65 and price_below_1d:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME (38.2 <= CHOP <= 61.8): Only trade strong signals
        else:
            # Only enter on very strong momentum in direction of 1d trend
            if hma_1d_slope > 0 and price_above_1w and roc > 8.0:
                desired_signal = SIZE_BASE
            elif hma_1d_slope < 0 and price_below_1w and roc < -8.0:
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