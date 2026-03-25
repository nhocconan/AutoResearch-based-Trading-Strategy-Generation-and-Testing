#!/usr/bin/env python3
"""
Experiment #1542: 4h Primary + 1d/1w HTF — KAMA Adaptive Trend with Regime Filter

Hypothesis: 4h timeframe offers optimal balance between signal quality and trade frequency.
KAMA (Kaufman Adaptive Moving Average) adjusts to volatility - fast in trends, slow in chop.
Combined with 1d HMA bias + Choppiness regime + loose RSI entries = consistent signals.

Key components:
1. 1d HMA(21) for major trend bias (never counter-trend against daily)
2. 1w HMA(10) for secular trend filter (reduce position size against weekly)
3. 4h KAMA(10,2,30) for adaptive trend following
4. Choppiness Index(14) regime: >61.8=range, <38.2=trend
5. RSI(14) loose entries: 35/65 thresholds (not 30/70 - too strict)
6. ATR(14) trailing stoploss at 2.5x
7. Discrete sizing: 0.0, ±0.25, ±0.30

Why this should beat current best (Sharpe=0.575):
- KAMA adapts to volatility better than HMA/EMA in crypto
- Dual HTF filter (1d + 1w) prevents major drawdowns
- Loose RSI thresholds guarantee ≥30 trades/train
- 4h TF = natural 30-50 trades/year (fee-efficient)
- Regime switch prevents trend strategies dying in chop

Entry logic (LOOSE to guarantee trades):
- LONG trend: 1d_HMA bullish + 1w_HMA neutral/bullish + CHOP<38 + KAMA bullish + RSI<65
- SHORT trend: 1d_HMA bearish + 1w_HMA neutral/bearish + CHOP<38 + KAMA bearish + RSI>35
- LONG range: CHOP>61 + RSI<40 (mean reversion)
- SHORT range: CHOP>61 + RSI>60 (mean reversion)

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adaptive_chop_regime_1d1w_v1"
timeframe = "4h"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adapts to market noise - fast in trends, slow in chop
    """
    n = len(close)
    if n < er_period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(er_period, n):
        if not np.isnan(close[i]) and not np.isnan(close[i - er_period]):
            price_change = abs(close[i] - close[i - er_period])
            volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
            if volatility > 1e-10:
                er[i] = price_change / volatility
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

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
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
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
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=10)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 4h indicators
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # KAMA slope (trend direction)
    kama_slope = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        if not np.isnan(kama_4h[i]) and not np.isnan(kama_4h[i-1]):
            kama_slope[i] = kama_4h[i] - kama_4h[i-1]
    
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
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_4h[i]) or np.isnan(kama_slope[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]):
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
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        is_trend_regime = chop < 38.2
        is_range_regime = chop > 61.8
        
        # === TREND DIRECTION (1d HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === SECULAR TREND (1w HMA) ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === KAMA TREND ===
        kama_bullish = kama_slope[i] > 0
        kama_bearish = kama_slope[i] < 0
        
        # === RSI ===
        rsi = rsi_14[i]
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # TREND REGIME: KAMA trend + 1d bias + 1w confirmation
        if is_trend_regime:
            # LONG: 1d bullish + KAMA bullish + RSI not overbought
            if price_above_1d and kama_bullish and rsi < 65:
                # Bonus if 1w also bullish
                if price_above_1w:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # SHORT: 1d bearish + KAMA bearish + RSI not oversold
            elif price_below_1d and kama_bearish and rsi > 35:
                # Bonus if 1w also bearish
                if price_below_1w:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # RANGE REGIME: RSI mean reversion (loose thresholds)
        elif is_range_regime:
            # LONG: RSI oversold
            if rsi < 40:
                desired_signal = SIZE_BASE
            
            # SHORT: RSI overbought
            elif rsi > 60:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: Only take signals with strong 1d confirmation
        else:
            # LONG: 1d bullish + KAMA bullish
            if price_above_1d and kama_bullish and rsi < 60:
                desired_signal = SIZE_BASE
            
            # SHORT: 1d bearish + KAMA bearish
            elif price_below_1d and kama_bearish and rsi > 40:
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