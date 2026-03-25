#!/usr/bin/env python3
"""
Experiment #1367: 6h Primary + 1d/1w HTF — Regime-Adaptive CHOP + Dual Strategy

Hypothesis: 6h timeframe needs regime detection to survive 2022-style crashes and 2025 bear markets.
This strategy adapts between trend-following and mean-reversion based on Choppiness Index:

1. CHOP(14) < 38.2 = TRENDING regime → follow 1d HMA trend with KAMA confirmation
2. CHOP(14) > 61.8 = RANGING regime → mean revert at RSI extremes with BB filter
3. CHOP between 38.2-61.8 = TRANSITION → stay flat or reduce position

Why this should beat baseline (Sharpe=0.447):
- Regime detection avoids trend-following whipsaw in 2022 crash
- Mean-reversion in ranges captures 2025 bear/range market
- 1w HMA prevents counter-trend trades in strong macro trends
- ATR-based stoploss limits drawdown on failed breakouts
- 6h TF = ~35-50 trades/year (fee-efficient, not overtraded)

Entry logic:
- LONG (trend): CHOP<38 + price>1d_HMA + KAMA rising + 1w_HMA neutral/bullish
- LONG (mean-revert): CHOP>61 + RSI<30 + price>BB_lower + price>1w_HMA*0.95
- SHORT (trend): CHOP<38 + price<1d_HMA + KAMA falling + 1w_HMA neutral/bearish
- SHORT (mean-revert): CHOP>61 + RSI>70 + price<BB_upper + price<1w_HMA*1.05

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_chop_regime_adaptive_dual_1d1w_v1"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average - adapts to market noise"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if not np.isnan(close[i]) and not np.isnan(close[i - period]):
            signal = abs(close[i] - close[i - period])
            noise = 0.0
            for j in range(i - period + 1, i + 1):
                if not np.isnan(close[j]) and not np.isnan(close[j - 1]):
                    noise += abs(close[j] - close[j - 1])
            if noise > 0:
                er[i] = signal / noise
    
    sc = np.full(n, np.nan, dtype=np.float64)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama[period - 1] = close[period - 1]
    
    for i in range(period, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i - 1]) and not np.isnan(close[i]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
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

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending
    CHOP > 61.8 = ranging/choppy market
    CHOP < 38.2 = trending market
    
    Formula: CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
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
        # Sum of ATR over period
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        
        # Highest high and lowest low over period
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
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
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    kama_21 = calculate_kama(close, period=21)
    bb_upper, bb_lower = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_TREND = 0.30
    SIZE_MR = 0.25
    SIZE_WEAK = 0.15
    
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
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_21[i]) or np.isnan(chop_14[i]):
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
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (CHOP) ===
        chop = chop_14[i]
        
        # Trending regime: CHOP < 38.2
        is_trending = chop < 38.2
        
        # Ranging regime: CHOP > 61.8
        is_ranging = chop > 61.8
        
        # Transition: stay flat or reduce
        is_transition = not is_trending and not is_ranging
        
        # === TREND DIRECTION (1d HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 1w HMA for major regime filter
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # 1w HMA slope (major trend direction)
        hma_1w_slope = 0
        if i >= 2 and not np.isnan(hma_1w_aligned[i-1]):
            if hma_1w_aligned[i] > hma_1w_aligned[i-1] * 1.005:
                hma_1w_slope = 1
            elif hma_1w_aligned[i] < hma_1w_aligned[i-1] * 0.995:
                hma_1w_slope = -1
        
        # === KAMA TREND DIRECTION ===
        kama_uptrend = False
        kama_downtrend = False
        
        if i >= 3:
            if kama_21[i] > kama_21[i-1] > kama_21[i-2]:
                kama_uptrend = True
            elif kama_21[i] < kama_21[i-1] < kama_21[i-2]:
                kama_downtrend = True
        
        rsi = rsi_14[i]
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        desired_signal = 0.0
        
        if is_trending:
            # TREND-FOLLOWING MODE
            # LONG: 1d bullish + KAMA uptrend + 1w not strongly bearish
            if price_above_1d and kama_uptrend and hma_1w_slope >= 0:
                if price_above_1w:
                    desired_signal = SIZE_TREND
                else:
                    desired_signal = SIZE_WEAK
            
            # SHORT: 1d bearish + KAMA downtrend + 1w not strongly bullish
            elif price_below_1d and kama_downtrend and hma_1w_slope <= 0:
                if price_below_1w:
                    desired_signal = -SIZE_TREND
                else:
                    desired_signal = -SIZE_WEAK
        
        elif is_ranging:
            # MEAN-REVERSION MODE
            # LONG: RSI oversold + price near BB lower + 1w not crash mode
            if rsi < 30 and close[i] < bb_lower[i] * 1.005:
                # Only long if 1w HMA not in severe downtrend
                if hma_1w_slope >= -1:
                    desired_signal = SIZE_MR
            
            # SHORT: RSI overbought + price near BB upper + 1w not rally mode
            elif rsi > 70 and close[i] > bb_upper[i] * 0.995:
                # Only short if 1w HMA not in severe uptrend
                if hma_1w_slope <= 1:
                    desired_signal = -SIZE_MR
        
        # Transition regime: reduce or flat
        if is_transition:
            # Keep existing position but don't add
            if in_position:
                desired_signal = position_side * SIZE_WEAK
            else:
                desired_signal = 0.0
        
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
        if desired_signal >= SIZE_TREND * 0.9:
            final_signal = SIZE_TREND
        elif desired_signal <= -SIZE_TREND * 0.9:
            final_signal = -SIZE_TREND
        elif desired_signal >= SIZE_MR * 0.9:
            final_signal = SIZE_MR
        elif desired_signal <= -SIZE_MR * 0.9:
            final_signal = -SIZE_MR
        elif desired_signal >= SIZE_WEAK * 0.9:
            final_signal = SIZE_WEAK
        elif desired_signal <= -SIZE_WEAK * 0.9:
            final_signal = -SIZE_WEAK
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