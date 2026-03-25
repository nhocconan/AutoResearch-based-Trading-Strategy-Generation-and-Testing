#!/usr/bin/env python3
"""
Experiment #1393: 5m Primary + 15m/4h HTF — Session-Filtered Trend Pullback

Hypothesis: 5m timeframe is completely unexplored (ZERO prior experiments). This strategy uses:
1. 4h HMA(21) for major trend bias (NEVER trade counter-trend on 5m)
2. 15m HMA(16) for intermediate trend confirmation
3. 5m RSI(3) extreme pullback entries (Connors-style mean reversion WITH trend)
4. Session filter: only trade 08:00-20:00 UTC (London/NY overlap = best liquidity)
5. Volume confirmation: taker_buy_volume ratio > 0.55 for longs, < 0.45 for shorts
6. ATR-based trailing stop (2.5x) to protect against 5m whipsaws

Why this should work where others failed:
- 4h trend filter prevents counter-trend deaths (2022 crash killer)
- Session filter avoids low-liquidity Asian hours (false breakouts)
- RSI(3) extreme entries catch pullbacks in established trends
- Volume confirmation ensures real participation, not fake moves
- 5m TF = 50-120 trades/year target (strict entry = not overtraded)

Entry logic:
- LONG: 4h_HMA bullish + 15m_HMA bullish + RSI(3) < 25 + volume_buy_ratio > 0.55 + session
- SHORT: 4h_HMA bearish + 15m_HMA bearish + RSI(3) > 75 + volume_buy_ratio < 0.45 + session

Target: Sharpe>0.5, trades>=50 train, trades>=5 test, DD>-35%
Timeframe: 5m
Size: 0.15-0.25 discrete (smaller due to higher trade frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_session_rsi3_pullback_hma_trend_15m4h_v1"
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

def calculate_volume_ratio(taker_buy_volume, volume):
    """Taker buy volume ratio (0-1, >0.5 = buying pressure)"""
    n = len(volume)
    ratio = np.full(n, np.nan, dtype=np.float64)
    mask = volume > 0
    ratio[mask] = taker_buy_volume[mask] / volume[mask]
    return ratio

def is_session_active(open_time):
    """
    Check if bar is within active trading session (08:00-20:00 UTC)
    open_time is in milliseconds since epoch
    """
    # Convert to hours UTC
    hours_utc = (open_time // (1000 * 60 * 60)) % 24
    # Active session: 08:00 to 20:00 UTC (London/NY overlap)
    return 8 <= hours_utc < 20

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_15m = get_htf_data(prices, '15m')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF indicators
    hma_15m_raw = calculate_hma(df_15m['close'].values, period=16)
    hma_15m_aligned = align_htf_to_ltf(prices, df_15m, hma_15m_raw)
    
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 5m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_3 = calculate_rsi(close, period=3)  # Fast RSI for pullback entries
    rsi_14 = calculate_rsi(close, period=14)  # Standard RSI for filter
    vol_ratio = calculate_volume_ratio(taker_buy_volume, volume)
    
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
    
    # Warmup period (need HTF alignment + indicators)
    min_bars = 150
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_3[i]) or np.isnan(rsi_14[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (MANDATORY for 5m) ===
        session_active = is_session_active(open_time[i])
        
        # === TREND DIRECTION (4h HMA major bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === INTERMEDIATE TREND (15m HMA confirmation) ===
        price_above_15m = close[i] > hma_15m_aligned[i]
        price_below_15m = close[i] < hma_15m_aligned[i]
        
        # === RSI PULLBACK CONDITIONS ===
        rsi_fast = rsi_3[i]
        rsi_std = rsi_14[i]
        
        # === VOLUME CONFIRMATION ===
        vol_buy_ratio = vol_ratio[i]
        
        # === ENTRY LOGIC (STRICT - limit to 50-120 trades/year) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + 15m bullish + RSI(3) extreme low + volume confirmation + session
        if price_above_4h and price_above_15m and rsi_fast < 25 and vol_buy_ratio > 0.55 and session_active:
            # Strong signal: RSI(14) also bullish (>45)
            if rsi_std > 45:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT: 4h bearish + 15m bearish + RSI(3) extreme high + volume confirmation + session
        elif price_below_4h and price_below_15m and rsi_fast > 75 and vol_buy_ratio < 0.45 and session_active:
            # Strong signal: RSI(14) also bearish (<55)
            if rsi_std < 55:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        
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