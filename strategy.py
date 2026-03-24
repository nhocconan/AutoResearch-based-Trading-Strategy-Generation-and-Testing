#!/usr/bin/env python3
"""
Experiment #005: 1h Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Volume/Session Filter

Hypothesis: After 4 failed experiments (all with Choppiness Index or overly complex filters),
this strategy SIMPLIFIES while using proven elements for lower timeframe:
1. 4h HMA(21) for trend direction - smooth, lag-reduced trend filter
2. 1h RSI(14) pullback entries - enter on dips in uptrend, rallies in downtrend
3. Volume confirmation (>0.8x 20-bar avg) - ensures real moves, not noise
4. Session filter (8-20 UTC) - avoids low-liquidity Asian night hours
5. ADX(14) > 20 - filters out choppy/ranging markets
6. ATR(14) 2.5x trailing stop - proven risk management

Why 1h with HTF works:
- 4h/1d determines SIGNAL DIRECTION (reduces whipsaw)
- 1h determines ENTRY TIMING (better entry price within HTF trend)
- Session + Volume filters keep trades to 30-60/year target (fee-efficient)
- Discrete sizing 0.25 minimizes fee churn on signal changes

Entry Logic:
- Long: 4h HMA bullish + 1h RSI 40-55 (pullback) + Volume > 0.8x avg + ADX > 20 + Session 8-20 UTC
- Short: 4h HMA bearish + 1h RSI 45-60 (rally) + Volume > 0.8x avg + ADX > 20 + Session 8-20 UTC
- Size: 0.25 (discrete, conservative for 1h TF)

Risk: 2.5x ATR trailing stop, signal→0 when stopped out
Target: Sharpe>0.3, trades>30/symbol train, >3/symbol test, DD>-40%, 30-60 trades/year
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - reduced lag trend indicator"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(series, span):
        if len(series) < span:
            return np.full(len(series), np.nan)
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_rsi(close, period=14):
    """RSI - momentum oscillator for pullback entries"""
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
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """ADX - trend strength filter"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10) * 100
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10) * 100
    
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_volume_avg(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = np.full(n, np.nan)
    for i in range(period - 1, n):
        vol_avg[i] = np.mean(volume[i - period + 1:i + 1])
    
    return vol_avg

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    ts_seconds = open_time / 1000.0
    utc_hour = pd.to_datetime(ts_seconds, unit='s').hour
    return utc_hour

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
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for stronger trend confirmation
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    vol_avg = calculate_volume_avg(volume, period=20)
    
    # Extract UTC hours for session filter
    utc_hours = np.array([get_utc_hour(ot) for ot in open_time])
    
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position size (conservative for 1h TF)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
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
        if np.isnan(rsi[i]) or np.isnan(adx[i]) or np.isnan(vol_avg[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        in_session = 8 <= utc_hours[i] <= 20
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > 0.8 * vol_avg[i]
        
        # === ADX TREND STRENGTH ===
        adx_ok = adx[i] > 20.0
        
        # === HTF TREND BIAS (4h + 1d HMA) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Stronger signal when both 4h and 1d agree
        strong_bull = hma_4h_bull and hma_1d_bull
        strong_bear = hma_4h_bear and hma_1d_bear
        
        # === RSI PULLBACK ENTRY (moderate thresholds for trade frequency) ===
        # Long: RSI pulled back to 40-55 in uptrend
        rsi_ok_long = 40.0 <= rsi[i] <= 55.0
        # Short: RSI rallied to 45-60 in downtrend
        rsi_ok_short = 45.0 <= rsi[i] <= 60.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Long entry: HTF bullish + RSI pullback + volume + ADX + session
        if strong_bull and rsi_ok_long and volume_ok and adx_ok and in_session:
            desired_signal = SIZE
        
        # Short entry: HTF bearish + RSI rally + volume + ADX + session
        elif strong_bear and rsi_ok_short and volume_ok and adx_ok and in_session:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals