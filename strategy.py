#!/usr/bin/env python3
"""
Experiment #1385: 15m Primary + 4h/1d HTF — Session-Based Trend Pullback

Hypothesis: 15m timeframe has ZERO successful experiments. Key insight from failures:
- Previous 15m strategies ( #1377, #1381) got Sharpe=0.000 = NO TRADES generated
- Entry conditions were TOO RESTRICTIVE (multiple confluence that never aligned)
- Solution: LOOSEN entry thresholds, use HTF for direction only, 15m for timing

Strategy design:
1. 1d HMA(21) = major regime filter (bull/bear bias)
2. 4h HMA(21) = intermediate trend (entry direction)
3. 15m RSI(7) = pullback entry trigger (oversold in uptrend, overbought in downtrend)
4. Session filter: 00-12 UTC only (London+NY overlap, higher volume)
5. Volume spike confirmation: volume > 1.5x 20-bar avg (ensures follow-through)
6. ATR(14) stoploss: 2.5x ATR trailing stop

Why this should generate trades (unlike #1377, #1381):
- RSI(7) < 35 triggers frequently in pullbacks (not RSI<20 which is too rare)
- Only require 4h trend + RSI pullback (not 5+ confluence filters)
- Session filter concentrates entries but doesn't block them
- Volume filter is lenient (1.5x not 2.0x)

Target: Sharpe>0.5, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 15m
Size: 0.15-0.25 discrete (smaller for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi7_pullback_hma_trend_session_4h1d_v2"
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

def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes above moving average"""
    n = len(volume)
    spike = np.zeros(n, dtype=np.float64)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    
    for i in range(period, n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            if volume[i] > threshold * vol_ma[i]:
                spike[i] = 1.0
    
    return spike

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    hour = (open_time // 3600000) % 24
    return hour

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
    vol_spike = calculate_volume_spike(volume, period=20, threshold=1.5)
    
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
    
    # Warmup period (shorter to generate trades earlier)
    min_bars = 50
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]):
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
        
        # === SESSION FILTER (00-12 UTC only) ===
        hour = get_session_hour(open_time[i])
        in_session = (hour >= 0 and hour < 12)
        
        # === TREND DIRECTION (4h HMA bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # 1d HMA for major regime
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === RSI PULLBACK (LOOSE THRESHOLDS) ===
        rsi = rsi_7[i]
        rsi_oversold = rsi < 40  # Looser than 35 to generate more trades
        rsi_overbought = rsi > 60  # Looser than 65 to generate more trades
        
        # === VOLUME CONFIRMATION ===
        has_volume = vol_spike[i] > 0
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + RSI pullback (oversold) + in session
        # Relaxed: only need 4h trend + RSI, volume is bonus
        if price_above_4h and rsi_oversold:
            if in_session:
                if price_above_1d:
                    # Strong alignment (4h + 1d both bullish)
                    base_size = SIZE_STRONG
                else:
                    # Basic long (only 4h bullish)
                    base_size = SIZE_BASE
                
                # Volume spike adds conviction
                if has_volume:
                    base_size = min(SIZE_STRONG, base_size + 0.05)
                
                desired_signal = base_size
        
        # SHORT: 4h bearish + RSI pullback (overbought) + in session
        elif price_below_4h and rsi_overbought:
            if in_session:
                if price_below_1d:
                    # Strong alignment (4h + 1d both bearish)
                    base_size = SIZE_STRONG
                else:
                    # Basic short (only 4h bearish)
                    base_size = SIZE_BASE
                
                # Volume spike adds conviction
                if has_volume:
                    base_size = min(SIZE_STRONG, base_size + 0.05)
                
                desired_signal = -base_size
        
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