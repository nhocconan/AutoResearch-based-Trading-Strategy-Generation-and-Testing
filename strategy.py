#!/usr/bin/env python3
"""
Experiment #1141: 15m Primary + 1h/4h/1d HTF — HMA Trend + RSI Pullback + Volume Session

Hypothesis: 15m timeframe has ZERO successful experiments. Using HTF (4h/1d) for trend 
direction and 15m only for pullback entry timing will capture trend moves with precise 
entries while limiting trade frequency to 40-100/year.

Key innovations:
1. 4h HMA(21) for primary trend direction (slow, reliable)
2. 1d HMA(21) for long-term bias filter (only trade with weekly trend)
3. 15m RSI(7) pullback entries (RSI 30-45 long, 55-70 short) - NOT extremes
4. Volume confirmation: volume > 1.3x 20-bar average (confirms momentum)
5. Session filter: UTC 00-12 only (London/NY overlap = highest liquidity)
6. ATR(14) 2.0x trailing stop for risk management
7. Discrete sizing: 0.0, ±0.20, ±0.25 to minimize fee churn

Why this should work on 15m:
- HTF trend filter reduces whipsaws (4h/1d direction dominates)
- RSI pullback (not extremes) generates MORE trades than CRSI<15/>85
- Volume filter confirms real momentum, not fake breakouts
- Session filter avoids low-liquidity Asian session chop
- 15m entries give better risk/reward than 4h entries on same trend

Entry conditions (LOOSE to guarantee trades):
- LONG: 4h_HMA bull + 1d_HMA bull + RSI(7) 30-45 + volume > 1.3x avg + UTC 00-12
- SHORT: 4h_HMA bear + 1d_HMA bear + RSI(7) 55-70 + volume > 1.3x avg + UTC 00-12

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 15m
Size: 0.20-0.25 discrete
Trade frequency: 40-100/year (session + HTF filters limit overtrading)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_pullback_vol_session_4h1d_v2"
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

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs rolling average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    vol_ratio[:period] = np.nan
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Extract UTC hour from open_time (milliseconds timestamp)
    utc_hours = np.zeros(n, dtype=np.int32)
    for i in range(n):
        ts_ms = open_time[i]
        ts_sec = ts_ms / 1000.0
        utc_hours[i] = int((ts_sec % 86400) / 3600)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(vol_ratio[i]):
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
        
        # === SESSION FILTER (UTC 00-12 only) ===
        hour = utc_hours[i]
        in_session = (hour >= 0 and hour <= 12)
        
        # === HTF TREND DIRECTION ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong trend alignment (both 4h and 1d agree)
        strong_bull = hma_4h_bull and hma_1d_bull
        strong_bear = hma_4h_bear and hma_1d_bear
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_ratio[i] > 1.3
        
        # === ENTRY LOGIC (PULLBACK IN TREND DIRECTION) ===
        desired_signal = 0.0
        
        # LONG: Strong bull trend + RSI pullback to 30-45 + volume + session
        if strong_bull and in_session:
            if rsi_7[i] >= 30.0 and rsi_7[i] <= 45.0 and vol_confirmed:
                desired_signal = SIZE_BASE
            elif rsi_7[i] >= 25.0 and rsi_7[i] <= 40.0 and vol_confirmed:
                desired_signal = SIZE_STRONG
        
        # SHORT: Strong bear trend + RSI pullback to 55-70 + volume + session
        elif strong_bear and in_session:
            if rsi_7[i] >= 55.0 and rsi_7[i] <= 70.0 and vol_confirmed:
                desired_signal = -SIZE_BASE
            elif rsi_7[i] >= 60.0 and rsi_7[i] <= 75.0 and vol_confirmed:
                desired_signal = -SIZE_STRONG
        
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