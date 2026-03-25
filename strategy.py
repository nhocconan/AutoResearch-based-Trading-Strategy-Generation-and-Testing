#!/usr/bin/env python3
"""
Experiment #1373: 5m Primary + 15m/1h HTF — Session-Filtered Trend Pullback

Hypothesis: 5m timeframe is completely unexplored (ZERO prior experiments). This strategy combines:
1. 1h HMA(21) for major trend bias (avoid counter-trend trades that kill Sharpe)
2. 15m RSI(14) for momentum confirmation (ensure trend has strength)
3. 5m RSI(7) pullback entries (buy dips in uptrend, sell rallies in downtrend)
4. Session filter: only trade 08-20 UTC (high volume hours, avoid Asian low-vol)
5. ATR-based stoploss (2.5x ATR trailing)

Why this should work where others failed:
- 1h trend filter prevents whipsaw (most 5m strategies fail here)
- Session filter avoids low-volume periods where spreads kill profits
- RSI pullback (not breakout) = better risk/reward in trends
- Small position size (0.15-0.25) accounts for higher trade frequency
- 5m entries with 1h trend = HTF trade frequency with LTF precision

Entry logic:
- LONG: price > 1h_HMA + 15m_RSI > 50 + 5m_RSI < 40 (pullback) + session active
- SHORT: price < 1h_HMA + 15m_RSI < 50 + 5m_RSI > 60 (rally) + session active

Target: Sharpe>0.5, trades>=50 train, trades>=5 test, DD>-35%
Timeframe: 5m
Size: 0.15-0.25 discrete (smaller due to more trades)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_rsi_pullback_hma_trend_session_15m1h_v1"
timeframe = "5m"
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

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def is_session_active(open_time, start_hour=8, end_hour=20):
    """Check if current bar is within active trading session (UTC)"""
    # open_time is in milliseconds since epoch
    hour = pd.to_datetime(open_time, unit='ms').hour
    return start_hour <= hour < end_hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_15m = get_htf_data(prices, '15m')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate and align HTF indicators
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    rsi_15m_raw = calculate_rsi(df_15m['close'].values, period=14)
    rsi_15m_aligned = align_htf_to_ltf(prices, df_15m, rsi_15m_raw)
    
    # Calculate 5m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_5m = calculate_rsi(close, period=7)  # Faster RSI for 5m entries
    ema_21 = calculate_ema(close, period=21)
    
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
    
    # Warmup period (need enough bars for all indicators)
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_5m[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1h_aligned[i]) or np.isnan(rsi_15m_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (only trade 08-20 UTC) ===
        session_active = is_session_active(open_time[i], start_hour=8, end_hour=20)
        
        if not session_active:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1h HMA bias) ===
        price_above_1h = close[i] > hma_1h_aligned[i]
        price_below_1h = close[i] < hma_1h_aligned[i]
        
        # === MOMENTUM CONFIRMATION (15m RSI) ===
        rsi_15m = rsi_15m_aligned[i]
        momentum_bullish = rsi_15m > 50
        momentum_bearish = rsi_15m < 50
        
        # === ENTRY SIGNAL (5m RSI pullback) ===
        rsi_5m_val = rsi_5m[i]
        
        # === VOL-ADJUSTED POSITION SIZING ===
        # Higher ATR = smaller position (risk management)
        if i > min_bars:
            atr_median = np.nanmedian(atr_14[min_bars:i])
            atr_ratio = atr_14[i] / atr_median if atr_median > 0 else 1.0
            vol_scale = 1.0 / max(0.5, min(2.0, atr_ratio))
        else:
            vol_scale = 1.0
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: 1h bullish + 15m momentum bullish + 5m RSI pullback (<40)
        if price_above_1h and momentum_bullish and rsi_5m_val < 40:
            # Strong signal if also above 5m EMA21
            if close[i] > ema_21[i]:
                base_size = SIZE_STRONG
            else:
                base_size = SIZE_BASE
            
            # Apply vol scaling
            desired_signal = base_size * vol_scale
        
        # SHORT: 1h bearish + 15m momentum bearish + 5m RSI rally (>60)
        elif price_below_1h and momentum_bearish and rsi_5m_val > 60:
            # Strong signal if also below 5m EMA21
            if close[i] < ema_21[i]:
                base_size = SIZE_STRONG
            else:
                base_size = SIZE_BASE
            
            # Apply vol scaling
            desired_signal = -base_size * vol_scale
        
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