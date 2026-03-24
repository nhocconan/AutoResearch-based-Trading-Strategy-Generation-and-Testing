#!/usr/bin/env python3
"""
Experiment #550: 1h Primary + 4h/1d HTF — HMA Trend + RSI Pullback

Hypothesis: After 472 failed experiments, the pattern is clear:
- Complex regime filters (CHOP + ADX) = 0 trades or negative Sharpe
- Lower TF (15m/30m) with session filters = 0 trades (Sharpe=0.000)
- SIMPLE trend following with HTF confirmation = best results

This strategy uses PROVEN patterns from best performers:
1. 4h HMA(21) for trend direction (not KAMA - HMA is faster, proven)
2. 1d HMA(21) for macro bias (align with higher TF)
3. 1h RSI(14) pullback entries (NOT extremes - 35-50 for longs in uptrend)
4. 1h HMA(21) for local trend confirmation
5. ATR(14)*2.5 stoploss only

Key differences from failed experiments:
- NO Choppiness filter (killed trade frequency in #538, #542, #548)
- NO ADX filter (too restrictive, caused whipsaws in #540, #544)
- NO session filter (caused 0 trades in #539, #541, #545, #549)
- Simpler entry logic = MORE trades (target 50-80/year on 1h)

Entry logic (simplified for trade frequency):
- Long: 4h HMA bullish + price > 4h HMA + 1h RSI 35-50 (pullback) + price > 1h HMA
- Short: 4h HMA bearish + price < 4h HMA + 1h RSI 50-65 (pullback) + price < 1h HMA

This catches pullbacks in established trends - high win rate, moderate frequency.
Position sizing: 0.25 base, 0.30 strong confluence. MAX 0.35.

Target: Sharpe>0.40 (beat current best 0.399), trades>=150 train, trades>=15 test
Timeframe: 1h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

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
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1h indicators
    hma_1h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1h[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (4h + 1d alignment) ===
        htf_bull = close[i] > hma_4h_aligned[i] and close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i] and close[i] < hma_1d_aligned[i]
        
        # === 4H HMA SLOPE (trend momentum) ===
        hma_4h_slope_bull = False
        hma_4h_slope_bear = False
        if i >= 5 and not np.isnan(hma_4h_aligned[i-5]):
            hma_4h_slope_bull = hma_4h_aligned[i] > hma_4h_aligned[i-5]
            hma_4h_slope_bear = hma_4h_aligned[i] < hma_4h_aligned[i-5]
        
        # === 1H LOCAL TREND ===
        ltf_bull = close[i] > hma_1h[i]
        ltf_bear = close[i] < hma_1h[i]
        
        # === RSI PULLBACK ZONES (not extremes!) ===
        # In uptrend: buy pullbacks (RSI 35-50)
        rsi_pullback_long = 35.0 <= rsi[i] <= 50.0
        # In downtrend: sell rallies (RSI 50-65)
        rsi_pullback_short = 50.0 <= rsi[i] <= 65.0
        
        # RSI momentum (improving)
        rsi_improving_long = False
        rsi_improving_short = False
        if i > 0 and not np.isnan(rsi[i-1]):
            rsi_improving_long = rsi[i] > rsi[i-1] and rsi[i] < 50.0
            rsi_improving_short = rsi[i] < rsi[i-1] and rsi[i] > 50.0
        
        # === ENTRY LOGIC (simplified for trade frequency) ===
        desired_signal = 0.0
        confluence_count = 0
        
        # LONG entries
        if htf_bull:
            # Base long: HTF bull + LTF bull + RSI pullback
            if ltf_bull and rsi_pullback_long:
                confluence_count = 2
                if hma_4h_slope_bull:
                    confluence_count += 1
                if rsi_improving_long:
                    confluence_count += 1
                
                if confluence_count >= 3:
                    desired_signal = SIZE_STRONG
                elif confluence_count >= 2:
                    desired_signal = SIZE_BASE
        
        # SHORT entries
        elif htf_bear:
            # Base short: HTF bear + LTF bear + RSI pullback
            if ltf_bear and rsi_pullback_short:
                confluence_count = 2
                if hma_4h_slope_bear:
                    confluence_count += 1
                if rsi_improving_short:
                    confluence_count += 1
                
                if confluence_count >= 3:
                    desired_signal = -SIZE_STRONG
                elif confluence_count >= 2:
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
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