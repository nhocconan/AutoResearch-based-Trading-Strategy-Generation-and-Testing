#!/usr/bin/env python3
"""
Experiment #1113: 5m Primary + 15m/4h HTF — Session-Filtered Trend Pullback

Hypothesis: 5m timeframe requires extreme selectivity to avoid fee drag. Using 4h HMA for
primary trend direction + 15m RSI for pullback entries + session filter (high liquidity hours)
will generate quality trades with controlled frequency (50-100 trades/year target).

Key innovations:
1. 4h HMA(21) for primary trend bias — only trade in HTF direction
2. 15m RSI(7) for pullback detection — faster than RSI14 for 5m entries
3. Session filter: 06:00-22:00 UTC only (avoids low-liquidity Asian overnight)
4. 5m ATR(14) for dynamic stoploss (2.0x ATR trailing)
5. Discrete sizing: 0.15 base, 0.20 strong (small due to high trade frequency)
6. Volume confirmation: taker_buy_volume ratio > 0.45 for long, < 0.55 for short

Why this should work:
- 5m has ZERO prior experiments — fresh approach needed
- HTF trend filter prevents counter-trend trades (major source of losses)
- Session filter reduces trades during low-liquidity periods (fewer whipsaws)
- RSI(7) is faster than RSI(14), better for 5m pullback detection
- Small position size (0.15-0.20) accounts for higher fee drag at 5m
- Volume confirmation adds confluence without being too restrictive

Entry conditions (LOOSE enough to guarantee trades):
- LONG: 4h_HMA_bull + 15m_RSI<45 + session_active + volume_confirm
- SHORT: 4h_HMA_bear + 15m_RSI>55 + session_active + volume_confirm
- Exit: RSI>65 (long) / RSI<35 (short) OR 2x ATR stoploss

Target: Sharpe>0.45, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 5m
Size: 0.15-0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_session_rsi_pullback_4h15m_trend_v1"
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

def is_session_active(open_time):
    """
    Check if timestamp is within high-liquidity session (06:00-22:00 UTC)
    open_time is in milliseconds since epoch
    """
    # Convert to hours UTC
    hour_utc = (open_time // (1000 * 60 * 60)) % 24
    return 6 <= hour_utc < 22

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
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    rsi_15m_raw = calculate_rsi(df_15m['close'].values, period=7)
    rsi_15m_aligned = align_htf_to_ltf(prices, df_15m, rsi_15m_raw)
    
    # Calculate 5m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    
    # Volume ratio (taker buy / total volume)
    volume_ratio = np.divide(taker_buy_volume, volume, out=np.zeros_like(volume), where=volume != 0)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
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
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_15m_aligned[i]):
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
        
        # === SESSION FILTER (06:00-22:00 UTC only) ===
        session_active = is_session_active(open_time[i])
        
        # === HTF TREND BIAS (4h HMA) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # Only trade during active session
        if session_active:
            # LONG: 4h trend up + 15m RSI pullback + volume confirmation
            if hma_4h_bull and rsi_15m_aligned[i] < 45.0 and volume_ratio[i] > 0.45:
                # Stronger signal if RSI very oversold
                if rsi_15m_aligned[i] < 35.0:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # SHORT: 4h trend down + 15m RSI pullback + volume confirmation
            elif hma_4h_bear and rsi_15m_aligned[i] > 55.0 and volume_ratio[i] < 0.55:
                # Stronger signal if RSI very overbought
                if rsi_15m_aligned[i] > 65.0:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
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
        
        # === EXIT CONDITIONS (RSI extremes) ===
        if in_position and position_side > 0 and rsi_7[i] > 65.0:
            desired_signal = 0.0  # Take profit on long
        
        if in_position and position_side < 0 and rsi_7[i] < 35.0:
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