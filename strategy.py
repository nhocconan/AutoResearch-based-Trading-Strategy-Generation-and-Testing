#!/usr/bin/env python3
"""
Experiment #1153: 5m Primary + 15m/4h HTF — Trend-Following Pullback with Session Filter

Hypothesis: 5m timeframe is unexplored territory. Using 4h HMA for trend direction + 15m RSI 
for pullback entries + session filter (08-20 UTC high volume) will capture intraday momentum
while avoiding noise and low-volume whipsaws.

Key innovations:
1. 4h HMA(21) for primary trend bias — ONLY trade with HTF trend
2. 15m RSI(14) pullback entries — long on RSI 35-45 dip in uptrend, short on RSI 55-65 rally in downtrend
3. Session filter: 08-20 UTC only (avoid Asian low-volume chop)
4. 5m ATR(14) 2.5x trailing stop for tight risk management
5. Small position size (0.15) due to higher trade frequency on 5m
6. Volume confirmation: only enter when 5m volume > 0.8 * 20-bar avg

Why this should work:
- 4h trend filter prevents counter-trend trades (major failure mode on lower TF)
- RSI pullback entries catch continuation moves with better risk/reward
- Session filter avoids 60% of noise (low-volume hours)
- 5m entries give precise timing within 4h trend structure
- Small size (0.15) handles fee drag from more frequent trades

Target: Sharpe>0.45, trades>=50/symbol train, trades>=5/symbol test, DD>-35%
Timeframe: 5m
Size: 0.15 (discrete: 0.0, ±0.15, ±0.25)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_hma_rsi_pullback_session_4h15m_v1"
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
    df_15m = get_htf_data(prices, '15m')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    rsi_15m_raw = calculate_rsi(df_15m['close'].values, period=14)
    rsi_15m_aligned = align_htf_to_ltf(prices, df_15m, rsi_15m_raw)
    
    # Calculate 5m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(rsi_15m_aligned[i]):
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
        
        # === SESSION FILTER (08-20 UTC only) ===
        # Convert open_time (ms) to hour
        timestamp_ms = open_time[i]
        hour_utc = (timestamp_ms // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === VOLUME FILTER ===
        vol_ok = not np.isnan(vol_sma[i]) and volume[i] > 0.8 * vol_sma[i]
        
        # === 4h TREND BIAS ===
        trend_bull = close[i] > hma_4h_aligned[i]
        trend_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m RSI PULLBACK SIGNALS ===
        # Long: RSI dipped to 35-45 in uptrend (pullback entry)
        rsi_long_pullback = 35.0 <= rsi_15m_aligned[i] <= 48.0
        # Short: RSI rallied to 55-65 in downtrend (pullback entry)
        rsi_short_pullback = 52.0 <= rsi_15m_aligned[i] <= 65.0
        
        # === 5m RSI CONFIRMATION (avoid catching falling knife) ===
        rsi_5m_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_5m_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG entry: 4h uptrend + 15m RSI pullback + session + volume + 5m RSI turning up
        if trend_bull and rsi_long_pullback and in_session and vol_ok and rsi_5m_rising:
            desired_signal = SIZE_BASE
        
        # Stronger long: deeper pullback or strong momentum
        if trend_bull and 30.0 <= rsi_15m_aligned[i] <= 40.0 and in_session and vol_ok:
            desired_signal = SIZE_STRONG
        
        # SHORT entry: 4h downtrend + 15m RSI pullback + session + volume + 5m RSI turning down
        if trend_bear and rsi_short_pullback and in_session and vol_ok and rsi_5m_falling:
            desired_signal = -SIZE_BASE
        
        # Stronger short: deeper pullback or strong momentum
        if trend_bear and 60.0 <= rsi_15m_aligned[i] <= 70.0 and in_session and vol_ok:
            desired_signal = -SIZE_STRONG
        
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