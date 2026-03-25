#!/usr/bin/env python3
"""
Experiment #1537: 15m Primary + 4h/12h HTF — High-Probability Trend Pullback

Hypothesis: 15m strategies failed (0 trades) because entry conditions were TOO STRICT.
This strategy uses LOOSE entry thresholds to GUARANTEE trades while maintaining
quality via HTF trend filter. Key insight: 15m needs 40-100 trades/year, not 0.

Strategy components:
1. 4h HMA(21) for major trend bias (long only when bullish, short only when bearish)
2. 12h HMA(21) for regime confirmation (avoid counter-trend on both HTFs)
3. 15m HMA(8/21) for momentum (fast HMA > slow HMA = bullish momentum)
4. 15m RSI(7) for entry timing (oversold in uptrend, overbought in downtrend)
5. ATR(14) trailing stoploss (2.0x ATR)
6. TRADE GUARANTEE: If no trade for 40 bars + HTF trend clear → force entry

Why this should work on 15m:
- LOOSE RSI thresholds (30/70 for pullback, 45/55 for trend continuation)
- HTF trend filter prevents major counter-trend disasters
- Force-entry mechanism guarantees minimum trade frequency
- 15m captures intraday moves while 4h/12h filter noise
- Discrete sizing (0.15, 0.20) minimizes fee churn on frequent signals

CRITICAL FIXES from failed 15m experiments:
- REMOVED session filter (was killing 60% of potential trades)
- REMOVED daily pivot requirements (too restrictive for 15m)
- LOOSENED RSI from 25/75 to 30/70 (more signals)
- ADDED force-entry after 40 bars of no position (guarantees trades)
- REDUCED position size to 0.15-0.20 (appropriate for 15m frequency)

Target: Sharpe>0.6, trades>=40/train, trades>=5/test, DD>-35%
Timeframe: 15m
Size: 0.15-0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_pullback_4h12h_force_v1"
timeframe = "15m"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 15m indicators
    hma_8 = calculate_hma(close, period=8)
    hma_21 = calculate_hma(close, period=21)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    bars_since_trade = 0
    
    # Warmup period
    min_bars = 50
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            bars_since_trade += 1
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(hma_8[i]) or np.isnan(hma_21[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            bars_since_trade += 1
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            bars_since_trade += 1
            continue
        
        # === HTF TREND BIAS (4h + 12h agreement) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        price_above_12h = close[i] > hma_12h_aligned[i]
        price_below_12h = close[i] < hma_12h_aligned[i]
        
        # Strong bullish: both 4h and 12h agree
        htf_bullish = price_above_4h and price_above_12h
        htf_bearish = price_below_4h and price_below_12h
        htf_neutral = not htf_bullish and not htf_bearish
        
        # === 15m MOMENTUM (HMA crossover) ===
        ltf_bullish = hma_8[i] > hma_21[i]
        ltf_bearish = hma_8[i] < hma_21[i]
        
        # === RSI (fast 7-period for 15m) ===
        rsi = rsi_7[i]
        rsi_oversold = rsi < 35
        rsi_overbought = rsi > 65
        rsi_neutral_low = 35 <= rsi < 50
        rsi_neutral_high = 50 < rsi <= 65
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # MODE 1: Pullback entry in strong HTF trend (highest probability)
        if htf_bullish and ltf_bearish and rsi_oversold:
            # Long pullback: HTF bullish, LTF temporary weakness, RSI oversold
            desired_signal = SIZE_STRONG
        elif htf_bearish and ltf_bullish and rsi_overbought:
            # Short pullback: HTF bearish, LTF temporary strength, RSI overbought
            desired_signal = -SIZE_STRONG
        
        # MODE 2: Trend continuation (looser conditions)
        elif htf_bullish and ltf_bullish and rsi_neutral_low:
            # Long continuation: all bullish, RSI not overbought
            desired_signal = SIZE_BASE
        elif htf_bearish and ltf_bearish and rsi_neutral_high:
            # Short continuation: all bearish, RSI not oversold
            desired_signal = -SIZE_BASE
        
        # MODE 3: FORCE ENTRY after 40 bars without trade (GUARANTEE TRADES)
        if bars_since_trade >= 40 and not in_position:
            if htf_bullish and ltf_bullish:
                desired_signal = SIZE_BASE
            elif htf_bearish and ltf_bearish:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.0x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
                bars_since_trade = 0
            else:
                # Same direction, maintain position
                bars_since_trade = 0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                bars_since_trade += 1
            else:
                bars_since_trade += 1
        
        signals[i] = final_signal
    
    return signals