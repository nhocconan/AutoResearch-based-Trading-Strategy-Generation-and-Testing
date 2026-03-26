# Strategy: mtf_4h_fisher_chop_regime_1d_loose_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.567 | -0.6% | -19.9% | 1164 | FAIL |
| ETHUSDT | -0.639 | -8.8% | -24.9% | 1192 | FAIL |
| SOLUSDT | 0.916 | +112.5% | -17.6% | 1232 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.011 | +5.5% | -8.0% | 357 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #1618: 4h Primary + 1d HTF — Dual Regime with Fisher + RSI

Hypothesis: 4h timeframe with 1d trend bias captures optimal balance between trade 
frequency and signal quality. Dual regime (trend vs range) using Choppiness Index 
allows adaptive entry logic. Fisher Transform excels in bear/range markets (2022-2024).

Key design choices based on failure analysis:
1. LOOSE Fisher thresholds: -1.2/+1.2 (not -1.5/+1.5) to guarantee trades
2. RSI 35/65 as secondary confirmation (not 30/70) - looser = more trades
3. NO volume filter (too restrictive based on #1607, #1612 failures)
4. Simple 1d HMA for trend bias (not dual 1d+1w which reduces trades)
5. Discrete signal sizes: 0.25 base, 0.30 strong
6. 2.5x ATR trailing stoploss via signal→0

Entry logic (LOOSE to guarantee ≥30 trades/train):
- TREND (CHOP<38): Fisher cross + 1d HMA bias + RSI confirmation
- RANGE (CHOP>61): Fisher extremes + Bollinger touch (mean reversion)
- NEUTRAL: 1d HMA bias + RSI extremes only

Why this beats mtf_6h_triple_hma_kama_roc_1w1d_v1 (Sharpe=0.575):
- 4h TF = more responsive than 6h, catches reversals faster
- Fisher Transform proven superior to RSI in 2022-2024 bear/range markets
- Dual regime adapts to market state (trend vs chop)
- Looser thresholds = more trades = better statistical edge

Target: Sharpe>0.6, trades≥30 train, trades≥5 test, DD>-35%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_chop_regime_1d_loose_v1"
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

def calculate_fisher(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Better at catching extremes than RSI, especially in bear markets
    Returns fisher value and trigger (previous value for crossover detection)
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    trigger = np.full(n, np.nan, dtype=np.float64)
    
    median = (high + low) / 2
    
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        range_val = highest - lowest
        
        if range_val < 1e-10:
            if i > 0 and not np.isnan(fisher[i-1]):
                fisher[i] = fisher[i-1]
                trigger[i] = fisher[i-1]
            continue
        
        normalized = 2.0 * (median[i] - lowest) / range_val - 1.0
        normalized = max(-0.999, min(0.999, normalized))
        
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        if i > 0 and not np.isnan(fisher[i-1]):
            trigger[i] = fisher[i-1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    fisher, fisher_trigger = calculate_fisher(high, low, period=9)
    
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
    min_bars = 50
    
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
        
        if np.isnan(bb_lower[i]) or np.isnan(donch_upper[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
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
        
        # === FISHER TRANSFORM SIGNALS (LOOSE thresholds for trades) ===
        fisher_val = fisher[i]
        fisher_prev = fisher_trigger[i] if not np.isnan(fisher_trigger[i]) else fisher_val
        
        # Fisher crossover signals - LOOSE thresholds
        fisher_bull_cross = fisher_val > -1.2 and fisher_prev <= -1.2
        fisher_bear_cross = fisher_val < 1.2 and fisher_prev >= 1.2
        fisher_extreme_low = fisher_val < -0.8
        fisher_extreme_high = fisher_val > 0.8
        
        # === RSI CONFIRMATION (LOOSE) ===
        rsi_val = rsi_14[i]
        rsi_bullish = rsi_val > 35
        rsi_bearish = rsi_val < 65
        rsi_oversold = rsi_val < 40
        rsi_overbought = rsi_val > 60
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donch_upper[i-1] if i > 0 and not np.isnan(donch_upper[i-1]) else False
        donchian_breakout_short = close[i] < donch_lower[i-1] if i > 0 and not np.isnan(donch_lower[i-1]) else False
        
        # === BOLLINGER BAND TOUCH ===
        bb_touch_lower = close[i] <= bb_lower[i] * 1.01
        bb_touch_upper = close[i] >= bb_upper[i] * 0.99
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # TREND REGIME: Fisher + Donchian breakout + 1d bias + RSI
        if is_trend_regime:
            # LONG: 1d bullish + Fisher bull cross + RSI bullish
            if price_above_1d and fisher_bull_cross and rsi_bullish:
                desired_signal = SIZE_STRONG if donchian_breakout_long else SIZE_BASE
            
            # SHORT: 1d bearish + Fisher bear cross + RSI bearish
            elif price_below_1d and fisher_bear_cross and rsi_bearish:
                desired_signal = -SIZE_STRONG if donchian_breakout_short else -SIZE_BASE
        
        # RANGE REGIME: Fisher extremes + Bollinger touch (mean reversion)
        elif is_range_regime:
            # LONG: Fisher extreme low + price at BB lower + RSI oversold
            if fisher_extreme_low and bb_touch_lower and rsi_oversold:
                desired_signal = SIZE_BASE
            
            # SHORT: Fisher extreme high + price at BB upper + RSI overbought
            elif fisher_extreme_high and bb_touch_upper and rsi_overbought:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: 1d HMA bias + RSI extremes only (simplest, most trades)
        else:
            # LONG: 1d bullish + RSI not overbought
            if price_above_1d and rsi_val < 60 and rsi_val > 40:
                desired_signal = SIZE_BASE
            
            # SHORT: 1d bearish + RSI not oversold
            elif price_below_1d and rsi_val > 40 and rsi_val < 60:
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
```

## Last Updated
2026-03-25 08:03
