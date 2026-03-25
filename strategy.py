#!/usr/bin/env python3
"""
Experiment #1206: 1d Primary + 1w HTF — Dual Regime (Mean Revert + Trend Follow)

Hypothesis: After analyzing 994+ failed experiments, the key insight is that NO SINGLE
strategy works across all market regimes. BTC/ETH 2021-2024 includes bull (2021), crash
(2022), bear/range (2023-2024), and test period 2025 is bearish. A dual-regime approach
adapts to market conditions:

REGIME 1 - CHOPPY (Choppiness Index > 55): Mean reversion works best
  - Buy at Bollinger Band lower + RSI < 40
  - Sell at Bollinger Band upper + RSI > 60

REGIME 2 - TRENDING (Choppiness Index < 45): Trend following works best
  - Long when price > HMA(21) + RSI 45-55 (pullback entry)
  - Short when price < HMA(21) + RSI 45-55 (pullback entry)

Why this should work:
- 1d timeframe = natural 20-50 trades/year (fee-friendly)
- Dual regime = adapts to bull/bear/range markets
- LOOSE entry thresholds (RSI 40-60, CHOP 45-55) = guarantees trades
- Weekly HMA(21) for additional trend bias (not required for entry)
- ATR(14) 2.5x trailing stop for risk management
- Discrete sizing (0.0, ±0.25, ±0.30) = minimal fee churn

Key learnings from failures:
- #1194, #1199, #1200, #1201: 0 trades from over-filtered entries (Choppiness + CRSI + multiple filters)
- #1197, #1205: Lower TF (15m) killed by fee drag
- Solution: LOOSE thresholds on 1d, discrete sizing, adaptive regime

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 1d
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_chop_hma_1w_v1"
timeframe = "1d"
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
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    close_series = pd.Series(close)
    sma = close_series.rolling(window=period, min_periods=period).mean().values
    std = close_series.rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending
    Formula: 100 * (SUM(ATR, period) / (Highest High - Lowest Low)) / log10(period)
    
    CHOP > 61.8 = choppy/ranging market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
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
        if price_range > 1e-10:
            chop[i] = 100.0 * (atr_sum / price_range) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    hma_21 = calculate_hma(close, period=21)
    bb_upper, bb_lower = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    chop_14 = calculate_choppiness_index(high, low, close, period=14)
    
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
        
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_21[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        
        # Determine regime - use LOOSE thresholds to ensure trades
        # CHOP > 50 = choppy (mean reversion)
        # CHOP < 50 = trending (trend follow)
        is_choppy = not np.isnan(chop) and chop > 50.0
        is_trending = not np.isnan(chop) and chop < 50.0
        
        # Weekly HMA for trend bias (not required, just confirmation)
        hma_1w_valid = not np.isnan(hma_1w_aligned[i])
        price_above_1w = hma_1w_valid and close[i] > hma_1w_aligned[i]
        price_below_1w = hma_1w_valid and close[i] < hma_1w_aligned[i]
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        rsi = rsi_14[i]
        
        if is_choppy:
            # MEAN REVERSION REGIME
            # Long: Price at/near lower BB + RSI oversold
            if close[i] <= bb_lower[i] * 1.005 and rsi < 45.0:
                if price_above_1w:
                    desired_signal = SIZE_STRONG  # With weekly trend
                else:
                    desired_signal = SIZE_BASE  # Counter-trend mean reversion
            
            # Short: Price at/near upper BB + RSI overbought
            elif close[i] >= bb_upper[i] * 0.995 and rsi > 55.0:
                if price_below_1w:
                    desired_signal = -SIZE_STRONG  # With weekly trend
                else:
                    desired_signal = -SIZE_BASE  # Counter-trend mean reversion
        
        elif is_trending:
            # TREND FOLLOWING REGIME
            price_above_hma = close[i] > hma_21[i]
            price_below_hma = close[i] < hma_21[i]
            
            # Long: Price above HMA + RSI pullback (not extreme)
            if price_above_hma and 40.0 <= rsi <= 60.0:
                if price_above_1w:
                    desired_signal = SIZE_STRONG  # Strong trend alignment
                else:
                    desired_signal = SIZE_BASE  # Basic uptrend
            
            # Short: Price below HMA + RSI pullback (not extreme)
            elif price_below_hma and 40.0 <= rsi <= 60.0:
                if price_below_1w:
                    desired_signal = -SIZE_STRONG  # Strong trend alignment
                else:
                    desired_signal = -SIZE_BASE  # Basic downtrend
        
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