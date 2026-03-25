#!/usr/bin/env python3
"""
Experiment #1308: 4h Primary + 12h/1d HTF — Dual Regime (Chop + Trend) + HMA + RSI

Hypothesis: Simple trend strategies fail in bear/range markets (2022 crash, 2025 bear).
This strategy uses Choppiness Index to detect regime and switches logic:
- CHOP > 61.8 (range): Mean reversion at Bollinger extremes + RSI filter
- CHOP < 38.2 (trend): Trend follow with HMA direction + momentum confirmation
- Between 38.2-61.8: Flat (no trades, avoid whipsaw)

Key differences from failed strategies:
1. Dual regime logic (not pure trend, not pure mean-revert)
2. 12h HMA for major trend bias (only trade with HTF direction)
3. 1d HMA for regime confirmation (extra filter to reduce whipsaw)
4. Choppiness Index (14) for regime detection - proven in literature
5. RSI(7) for entry timing (faster than RSI(14), catches reversals)
6. ATR(14) 2.5x trailing stop for risk management
7. LOOSE entry thresholds to guarantee 30-60 trades/year on 4h

Why this should work on BTC/ETH (not just SOL):
- Range logic captures 2022 bottom whipsaw and 2025 bear chop
- Trend logic captures 2021 bull and strong rallies
- HTF filters prevent counter-trend trades that destroy Sharpe
- 4h timeframe = natural 30-50 trades/year (fee-friendly)

Entry logic:
- LONG (range): CHOP>61.8 + price<BB_lower + RSI(7)<25 + 12h_HMA not falling hard
- LONG (trend): CHOP<38.2 + 12h_HMA rising + 1d_HMA bullish + price>12h_HMA
- SHORT (range): CHOP>61.8 + price>BB_upper + RSI(7)>75 + 12h_HMA not rising hard
- SHORT (trend): CHOP<38.2 + 12h_HMA falling + 1d_HMA bearish + price<12h_HMA

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_chop_hma_rsi_12h1d_v1"
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

def calculate_rsi(close, period=7):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures whether market is trending or choppy/ranging
    CHOP > 61.8 = choppy/range (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    
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
        atr_sum = np.sum(tr[i - period + 1:i + 1])
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
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    bb_upper, bb_lower, bb_sma = calculate_bollinger(close, period=20, std_mult=2.0)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Also calculate 4h HMA for local trend
    hma_4h = calculate_hma(close, period=21)
    
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
        
        if np.isnan(rsi_7[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
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
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        is_choppy = chop > 61.8
        is_trending = chop < 38.2
        # Between 38.2-61.8 = transitional (no trades)
        
        # === TREND DIRECTION (12h HMA slope + 1d HMA bias) ===
        hma_12h_slope = 0.0
        if i >= 3 and not np.isnan(hma_12h_aligned[i-3]):
            hma_12h_slope = hma_12h_aligned[i] - hma_12h_aligned[i-3]
        
        # 1d HMA bias
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 4h price vs 4h HMA for local confirmation
        price_above_4h = close[i] > hma_4h[i] if not np.isnan(hma_4h[i]) else False
        price_below_4h = close[i] < hma_4h[i] if not np.isnan(hma_4h[i]) else False
        
        # === RSI for mean reversion ===
        rsi = rsi_7[i]
        
        # === ENTRY LOGIC (DUAL REGIME) ===
        desired_signal = 0.0
        
        if is_choppy:
            # === MEAN REVERSION MODE ===
            # LONG: price at BB lower + RSI oversold + 12h HMA not strongly bearish
            if close[i] <= bb_lower[i] and rsi < 25:
                if hma_12h_slope > -0.02 * close[i]:  # Not strongly falling
                    if rsi < 20:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
            
            # SHORT: price at BB upper + RSI overbought + 12h HMA not strongly bullish
            elif close[i] >= bb_upper[i] and rsi > 75:
                if hma_12h_slope < 0.02 * close[i]:  # Not strongly rising
                    if rsi > 80:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
        
        elif is_trending:
            # === TREND FOLLOWING MODE ===
            # LONG: 12h HMA rising + 1d bullish + price above 12h HMA
            if hma_12h_slope > 0 and price_above_1d and price_above_4h:
                if price_above_1d:
                    desired_signal = SIZE_BASE
                if hma_12h_slope > 0.03 * close[i]:  # Strong uptrend
                    desired_signal = SIZE_STRONG
            
            # SHORT: 12h HMA falling + 1d bearish + price below 12h HMA
            elif hma_12h_slope < 0 and price_below_1d and price_below_4h:
                if price_below_1d:
                    desired_signal = -SIZE_BASE
                if hma_12h_slope < -0.03 * close[i]:  # Strong downtrend
                    desired_signal = -SIZE_STRONG
        
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