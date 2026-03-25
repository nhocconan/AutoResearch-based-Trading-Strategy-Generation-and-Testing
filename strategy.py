#!/usr/bin/env python3
"""
Experiment #1102: 4h Primary + 1d/1w HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: Previous regime-switching strategies (Chop + CRSI) had TOO MANY filters,
causing 0 trades. This version SIMPLIFIES: pure HMA trend following with RSI pullback
entries. Fewer confluence requirements = more trades while maintaining quality.

Key changes from failed #1002:
1. REMOVED Choppiness Index regime detection (caused whipsaw + filtered too many trades)
2. REMOVED Connors RSI (too complex, extreme thresholds = 0 trades on 4h)
3. SIMPLIFIED to: 1w HMA bias + 1d HMA confirmation + 4h RSI pullback entry
4. LOOSENED RSI thresholds: 35-65 range instead of CRSI extremes
5. Added volume confirmation (optional, not required) to filter false breakouts

Entry logic (MUCH LOOSER to guarantee trades):
- LONG: close > 1d_HMA > 1w_HMA + RSI(14) pullback to 40-55 + volume > 0.8*avg
- SHORT: close < 1d_HMA < 1w_HMA + RSI(14) rally to 45-60 + volume > 0.8*avg
- Exit: RSI crosses opposite threshold OR stoploss hit (2.5x ATR)

Why this should work:
- HMA reduces lag vs EMA (proven in literature)
- Multi-TF alignment (1w > 1d > 4h) ensures we trade with major trend
- RSI pullback entries catch retracements, not breakouts (higher win rate)
- 4h timeframe = 20-50 trades/year target (fee-efficient)
- Discrete sizing (0.0, ±0.25, ±0.30) minimizes fee churn

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_simplified_1d1w_v2"
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    hma_4h_21 = calculate_hma(close, period=21)
    hma_4h_48 = calculate_hma(close, period=48)
    vol_sma_20 = calculate_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_48[i]):
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
        
        # === HTF TREND BIAS (1w + 1d HMA alignment) ===
        # Bullish: price > 1d_HMA > 1w_HMA
        bull_bias = (close[i] > hma_1d_aligned[i]) and (hma_1d_aligned[i] > hma_1w_aligned[i])
        # Bearish: price < 1d_HMA < 1w_HMA
        bear_bias = (close[i] < hma_1d_aligned[i]) and (hma_1d_aligned[i] < hma_1w_aligned[i])
        
        # === 4h TREND CONFIRMATION (HMA crossover) ===
        hma_bull = hma_4h_21[i] > hma_4h_48[i]
        hma_bear = hma_4h_21[i] < hma_4h_48[i]
        
        # === VOLUME FILTER (optional, not strict) ===
        vol_ok = True
        if not np.isnan(vol_sma_20[i]) and vol_sma_20[i] > 0:
            vol_ok = volume[i] > 0.7 * vol_sma_20[i]  # 70% of avg volume
        
        # === ENTRY LOGIC (SIMPLIFIED - LOOSE THRESHOLDS) ===
        desired_signal = 0.0
        
        # LONG entry: HTF bull + 4h HMA bull + RSI pullback (not overbought)
        if bull_bias and hma_bull and vol_ok:
            # RSI pullback to 40-55 range (buying dip in uptrend)
            if 38.0 <= rsi_14[i] <= 58.0:
                desired_signal = SIZE_BASE
            # Stronger signal if RSI just crossed up from oversold
            elif rsi_14[i] < 45.0 and i > 0 and rsi_14[i-1] < rsi_14[i]:
                desired_signal = SIZE_STRONG
        
        # SHORT entry: HTF bear + 4h HMA bear + RSI rally (not oversold)
        elif bear_bias and hma_bear and vol_ok:
            # RSI rally to 42-62 range (selling rip in downtrend)
            if 42.0 <= rsi_14[i] <= 62.0:
                desired_signal = -SIZE_BASE
            # Stronger signal if RSI just crossed down from overbought
            elif rsi_14[i] > 55.0 and i > 0 and rsi_14[i-1] > rsi_14[i]:
                desired_signal = -SIZE_STRONG
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            if low[i] < trailing_stop:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            if high[i] > trailing_stop:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT CONDITIONS (RSI opposite extreme) ===
        if in_position and position_side > 0 and rsi_14[i] > 70.0:
            desired_signal = 0.0  # Take profit on long
        
        if in_position and position_side < 0 and rsi_14[i] < 30.0:
            desired_signal = 0.0  # Take profit on short
        
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
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals