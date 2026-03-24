#!/usr/bin/env python3
"""
Experiment #701: 15m Primary + 4h/1d HTF — HMA Trend + RSI(7) Extreme Entries

Hypothesis: 15m timeframe needs LOOSE entry conditions to generate trades (see 0-trade failures).
Using 4h HMA for trend bias (not 1d/1w which are too slow for 15m), 15m HMA(16/48) for local trend,
and RSI(7) for faster entry signals. RSI(7) reaches extremes more often than RSI(14).

Key innovations for 15m:
1. 4h HMA(21) bias - only one HTF filter (not 1d+1w which caused 0 trades)
2. RSI(7) instead of RSI(14) - reaches extremes 2-3x more often
3. Loose RSI thresholds: <40 for long, >60 for short (not <30/>70)
4. No session filter - session filters caused 0 trades in exp#689, #693, #696
5. No ADX/Choppiness - these killed trade generation in exp#691, #692, #698
6. ATR(14) trailing stop at 2.5x for risk management
7. Discrete sizing: 0.0, ±0.15, ±0.25 (smaller for 15m frequency)

Entry conditions (LOOSE to ensure trades):
- LONG: price > 4h HMA AND 15m HMA16 > HMA48 AND RSI(7) < 45
- SHORT: price < 4h HMA AND 15m HMA16 < HMA48 AND RSI(7) > 55
- No additional filters - simplicity ensures trade generation

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 15m
Size: 0.15-0.25 discrete (smaller than 12h due to higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi7_loose_4h_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
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
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    rs[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100.0
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
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
    
    # Calculate and align HTF HMA
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 15m indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    rsi_7 = calculate_rsi(close, period=7)  # Faster RSI for more signals
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.25
    
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
        
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or np.isnan(rsi_7[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === HMA CROSSOVER TREND (15m) ===
        hma_bull = hma_16[i] > hma_48[i]
        hma_bear = hma_16[i] < hma_48[i]
        
        # === RSI(7) EXTREME ENTRY (LOOSE THRESHOLDS) ===
        # Long on oversold in uptrend (RSI(7) < 45 is common)
        rsi_long = rsi_7[i] < 45.0
        # Short on overbought in downtrend (RSI(7) > 55 is common)
        rsi_short = rsi_7[i] > 55.0
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + 15m HMA bull + RSI(7) oversold
        if htf_4h_bull and hma_bull and rsi_long:
            desired_signal = SIZE_STRONG
        elif htf_4h_bull and hma_bull:
            # Weaker: just trend alignment without RSI
            desired_signal = SIZE_BASE
        elif htf_4h_bull and rsi_long:
            # Weaker: HTF bias + RSI without HMA cross
            desired_signal = SIZE_BASE
        
        # SHORT: 4h bearish + 15m HMA bear + RSI(7) overbought
        elif htf_4h_bear and hma_bear and rsi_short:
            desired_signal = -SIZE_STRONG
        elif htf_4h_bear and hma_bear:
            # Weaker: just trend alignment without RSI
            desired_signal = -SIZE_BASE
        elif htf_4h_bear and rsi_short:
            # Weaker: HTF bias + RSI without HMA cross
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
        elif abs(desired_signal) >= SIZE_BASE * 0.4:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.5
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