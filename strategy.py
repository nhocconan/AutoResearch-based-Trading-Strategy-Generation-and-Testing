#!/usr/bin/env python3
"""
Experiment #1077: 15m Primary + 4h/12h HTF — Simple Trend + RSI Pullback

Hypothesis: After 8 failed 15m experiments with Sharpe=0.000 (ZERO trades), 
this strategy uses LOOSE entry conditions to guarantee trade generation while
maintaining selectivity through HTF trend filter.

Key innovations:
1. 4h HMA(21) for primary trend bias — only trade in HTF trend direction
2. 15m RSI(7) for entry timing — oversold in uptrend, overbought in downtrend
3. 12h ATR ratio for volatility filter — avoid extreme vol spikes
4. Session bias: 00-12 UTC preferred (London/NY overlap)
5. Simple discrete sizing: 0.0, ±0.20, ±0.30
6. ATR(14) 2.5x trailing stoploss

Why this should work (learning from 15m failures):
- Previous 15m strategies (#1069, #1070, #1073, #1076) all got Sharpe=0.000 = NO TRADES
- Entry conditions were TOO STRICT (multiple confluence that never aligned)
- This uses LOOSE RSI thresholds (35/65 not 20/80) to guarantee entries
- HTF trend filter prevents counter-trend trades without blocking all signals
- Target: 60-100 trades/year on 15m (fee drag ~3-5%)

Entry conditions (LOOSE to guarantee trades):
- LONG: 4h_HMA_bull + 15m_RSI(7)<40 + vol_ok
- SHORT: 4h_HMA_bear + 15m_RSI(7)>60 + vol_ok
- No regime switching complexity (failed in #1068, #1070)

Target: Sharpe>0.45, trades>=40 train, trades>=5 test, DD>-40%
Timeframe: 15m
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_pullback_4h12h_simple_v1"
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
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    atr_12h_raw = calculate_atr(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, period=14)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h_raw)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Volume MA for filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
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
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(atr_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (4h HMA) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === VOLATILITY FILTER (12h ATR ratio) ===
        # Avoid trading during extreme volatility spikes
        vol_ok = True
        if atr_12h_aligned[i] > 0:
            atr_ratio = atr_14[i] / (atr_12h_aligned[i] / 4.0)  # Scale 12h ATR to 15m
            if atr_ratio > 2.5 or atr_ratio < 0.3:
                vol_ok = False
        
        # === VOLUME FILTER ===
        vol_filter = True
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            if volume[i] < vol_ma[i] * 0.5:
                vol_filter = False
        
        # === SESSION FILTER (prefer 00-12 UTC) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        session_preferred = (0 <= hour_utc <= 12)
        
        # === ENTRY LOGIC (LOOSE CONDITIONS TO GUARANTEE TRADES) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + RSI(7) oversold + filters
        if hma_4h_bull and vol_ok and vol_filter:
            if rsi_7[i] < 40:
                # Strong signal if RSI very oversold
                if rsi_7[i] < 30:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            # Also enter on RSI(14) pullback in strong trend
            elif rsi_14[i] < 45 and rsi_7[i] < 50:
                desired_signal = SIZE_BASE
        
        # SHORT: 4h bearish + RSI(7) overbought + filters
        elif hma_4h_bear and vol_ok and vol_filter:
            if rsi_7[i] > 60:
                # Strong signal if RSI very overbought
                if rsi_7[i] > 70:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            # Also enter on RSI(14) rally in strong downtrend
            elif rsi_14[i] > 55 and rsi_7[i] > 50:
                desired_signal = -SIZE_BASE
        
        # Session boost: increase size during preferred hours
        if session_preferred and desired_signal != 0:
            if desired_signal > 0:
                desired_signal = min(desired_signal * 1.2, SIZE_STRONG)
            else:
                desired_signal = max(desired_signal * 1.2, -SIZE_STRONG)
        
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