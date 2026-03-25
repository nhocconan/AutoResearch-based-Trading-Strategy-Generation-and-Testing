#!/usr/bin/env python3
"""
Experiment #1609: 15m Primary + 1h/4h HTF — RSI Mean Reversion with Trend Filter

Hypothesis: 15m timeframe has failed 3x in a row with 0 trades due to OVERLY STRICT
entry conditions. This strategy uses SIMPLE, LOOSE entry logic to GUARANTEE trades:

Key changes vs failed 15m attempts (#1597, #1601, #1605):
1. NO SESSION FILTER - crypto trades 24/7, session filters kill trades
2. LOOSE RSI thresholds: RSI(7) < 25 or > 75 (not < 20 or > 80)
3. SINGLE HTF filter: 1h HMA for trend direction only (not 4h + 1d + session)
4. SIMPLE exit: RSI crosses 50 OR ATR stoploss (not complex trailing)
5. DISCRETE sizing: 0.20 base, 0.25 strong (minimize fee churn)

Why this should work on 15m:
- RSI(7) on 15m = 1.75 hours lookback, catches intraday extremes
- 1h HMA(21) = trend filter without over-filtering
- Mean reversion works in 2025 bear/range market
- Target: 50-100 trades/year (0.14-0.27 trades/day)

Entry logic (LOOSE to guarantee ≥30 trades/train):
- LONG: 1h_HMA bullish (price > HMA) + RSI(7) < 25 + RSI rising
- SHORT: 1h_HMA bearish (price < HMA) + RSI(7) > 75 + RSI falling

Exit logic:
- RSI crosses back through 50 (mean reversion complete)
- ATR(14) * 2.5 stoploss (protect from trend continuation)

Timeframe: 15m
Size: 0.20-0.25 discrete
Target: Sharpe>0.5, trades>=40 train, trades>=5 test, DD>-35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi7_hma1h_meanrev_v1"
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

def calculate_rsi(close, period=7):
    """Relative Strength Index - period=7 for faster signals on 15m"""
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
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF indicators
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track RSI for exit signals
    prev_rsi = np.nan
    
    # Warmup period
    min_bars = 50
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_rsi = rsi_7[i] if not np.isnan(rsi_7[i]) else prev_rsi
            continue
        
        if np.isnan(rsi_7[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_rsi = rsi_7[i] if not np.isnan(rsi_7[i]) else prev_rsi
            continue
        
        if np.isnan(hma_1h_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_rsi = rsi_7[i] if not np.isnan(rsi_7[i]) else prev_rsi
            continue
        
        # === TREND DIRECTION (1h HMA bias) ===
        price_above_1h = close[i] > hma_1h_aligned[i]
        price_below_1h = close[i] < hma_1h_aligned[i]
        
        # === 4h REGIME FILTER (only for position sizing, not entry) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === RSI SIGNALS (LOOSE thresholds for trade frequency) ===
        rsi_val = rsi_7[i]
        rsi_prev = prev_rsi if not np.isnan(prev_rsi) else rsi_val
        
        # RSI extreme levels
        rsi_oversold = rsi_val < 25
        rsi_overbought = rsi_val > 75
        
        # RSI momentum (rising/falling)
        rsi_rising = rsi_val > rsi_prev
        rsi_falling = rsi_val < rsi_prev
        
        # RSI mean reversion exit signals
        rsi_cross_above_50 = rsi_val > 50 and rsi_prev <= 50
        rsi_cross_below_50 = rsi_val < 50 and rsi_prev >= 50
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 1h bullish + RSI oversold + RSI rising (momentum confirmation)
        if price_above_1h and rsi_oversold and rsi_rising:
            # Stronger size if 4h also bullish
            desired_signal = SIZE_STRONG if price_above_4h else SIZE_BASE
        
        # SHORT: 1h bearish + RSI overbought + RSI falling (momentum confirmation)
        elif price_below_1h and rsi_overbought and rsi_falling:
            # Stronger size if 4h also bearish
            desired_signal = -SIZE_STRONG if price_below_4h else -SIZE_BASE
        
        # === EXIT LOGIC ===
        # Close position if RSI mean-reverts through 50
        if in_position and position_side > 0 and rsi_cross_above_50:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_cross_below_50:
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
        prev_rsi = rsi_val
    
    return signals