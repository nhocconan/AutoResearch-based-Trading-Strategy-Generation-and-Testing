#!/usr/bin/env python3
"""
Experiment #987: 6h Primary + 1d/1w HTF — HMA Trend + RSI Pullback

Hypothesis: Simplified 6h strategy using proven HMA trend + RSI pullback pattern
will outperform complex regime-based strategies that generate too few trades.

Key innovations:
1. 1d HMA(21) for intermediate trend bias (proven on 4h strategies)
2. 1w momentum (close > open) for weekly directional bias
3. 6h HMA(16/48) dual for primary trend confirmation
4. RSI(14) pullback entries: RSI<45 in uptrend, RSI>55 in downtrend
5. NO CHOP filter (eliminated many valid signals in #955)
6. ATR(14) 2.5x trailing stop for risk management

Why this should work:
- Simpler entry conditions = MORE trades (fixes #955's 0-trade problem)
- HMA trend + RSI pullback is proven on 4h (mtf_hma_rsi_zscore_v1 Sharpe=5.4)
- 6h captures multi-day swings without 4h noise
- Looser RSI thresholds (45/55 vs 40/60) ensure sufficient trade frequency
- HTF bias prevents counter-trend trades in strong moves

Entry conditions (LOOSE to guarantee 30+ trades):
- LONG = 1w bull + 1d HMA bull + 6h HMA bull + RSI(14) < 45
- SHORT = 1w bear + 1d HMA bear + 6h HMA bear + RSI(14) > 55
- Exit = RSI crosses 50 (mean reversion) or stoploss hit

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_hma_rsi_pullback_1d1w_v2"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - proven trend indicator"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
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
    """Relative Strength Index for pullback detection"""
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Weekly momentum: close vs open (bullish if close > open)
    weekly_momentum_raw = (df_1w['close'].values - df_1w['open'].values) / (df_1w['open'].values + 1e-10)
    weekly_momentum_aligned = align_htf_to_ltf(prices, df_1w, weekly_momentum_raw)
    
    # Calculate 6h indicators
    hma_6h_16 = calculate_hma(close, period=16)
    hma_6h_48 = calculate_hma(close, period=48)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
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
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_6h_16[i]) or np.isnan(hma_6h_48[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(weekly_momentum_aligned[i]):
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
        
        # === HTF BIAS (1w momentum + 1d HMA) ===
        htf_1w_bull = weekly_momentum_aligned[i] > 0.0
        htf_1w_bear = weekly_momentum_aligned[i] < 0.0
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 6h TREND (HMA 16/48) ===
        hma_6h_bull = hma_6h_16[i] > hma_6h_48[i]
        hma_6h_bear = hma_6h_16[i] < hma_6h_48[i]
        
        # === RSI PULLBACK (LOOSE THRESHOLDS FOR MORE TRADES) ===
        rsi_oversold = rsi_14[i] < 45  # Pullback in uptrend
        rsi_overbought = rsi_14[i] > 55  # Rally in downtrend
        
        # === RSI EXIT SIGNALS ===
        rsi_neutral = 45 <= rsi_14[i] <= 55
        
        # === ENTRY LOGIC (SIMPLIFIED - MORE TRADES) ===
        desired_signal = 0.0
        
        # LONG entries: All HTF bullish + RSI pullback
        if htf_1w_bull and htf_1d_bull and hma_6h_bull and rsi_oversold:
            desired_signal = SIZE_STRONG
        elif htf_1w_bull and hma_6h_bull and rsi_oversold:
            # Weaker condition: just 6h + 1w bull
            desired_signal = SIZE_BASE
        
        # SHORT entries: All HTF bearish + RSI rally
        elif htf_1w_bear and htf_1d_bear and hma_6h_bear and rsi_overbought:
            desired_signal = -SIZE_STRONG
        elif htf_1w_bear and hma_6h_bear and rsi_overbought:
            # Weaker condition: just 6h + 1w bear
            desired_signal = -SIZE_BASE
        
        # === EXIT LOGIC (RSI mean reversion) ===
        if in_position:
            if position_side > 0 and rsi_neutral:
                # Long position: exit when RSI recovers to neutral
                desired_signal = 0.0
            elif position_side < 0 and rsi_neutral:
                # Short position: exit when RSI drops to neutral
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