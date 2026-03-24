#!/usr/bin/env python3
"""
Experiment #673: 5m Primary + 15m/4h HTF — Session RSI Mean Reversion with Trend Filter

Hypothesis: 5m timeframe is untested territory. Key insight: 5m needs EXTREME selectivity.
Use 4h HMA for trend direction (only trade WITH trend), 15m RSI for entry timing
(mean reversion in direction of trend), session filter for liquidity (08-20 UTC),
and volume confirmation to avoid fake breakouts.

Why this might work on 5m:
1. 4h HMA(21) = strong trend filter (only long when price > 4h HMA)
2. 15m RSI(7) = faster RSI for 5m entries (oversold < 35 in uptrend, overbought > 65 in downtrend)
3. Session filter 08-20 UTC = avoid low liquidity Asian night hours
4. Volume > SMA20 = confirm real moves, not noise
5. ATR(14) 2.5x trailing stop = protect from 5m whipsaws
6. Small size (0.15) = many trades but controlled fee drag

Entry conditions (LOOSE to ensure >=10 trades/symbol):
- LONG: 4h HMA bull + 15m RSI < 40 + session active + volume ok
- SHORT: 4h HMA bear + 15m RSI > 60 + session active + volume ok

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-50%
Timeframe: 5m
Size: 0.15 (small due to higher trade frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_session_rsi_4h15m_trend_v1"
timeframe = "5m"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0.0)
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_sma(values, period):
    """Simple Moving Average"""
    n = len(values)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(values).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_15m = get_htf_data(prices, '15m')
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 15m RSI for entry timing
    rsi_15m_raw = calculate_rsi(df_15m['close'].values, period=7)
    rsi_15m_aligned = align_htf_to_ltf(prices, df_15m, rsi_15m_raw)
    
    # Calculate 5m indicators
    atr = calculate_atr(high, low, close, period=14)
    volume_sma = calculate_sma(volume, 20)
    
    signals = np.zeros(n)
    SIZE = 0.15  # Small size for 5m (many trades = fee drag)
    
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
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_15m_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(volume_sma[i]) or volume_sma[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        # Extract hour from open_time (milliseconds timestamp)
        open_time_ms = prices["open_time"].iloc[i]
        hour_utc = (open_time_ms // 3600000) % 24
        session_active = 8 <= hour_utc <= 20
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > volume_sma[i] * 0.8  # At least 80% of avg volume
        
        # === 4H TREND DIRECTION (HTF BIAS) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 15M RSI ENTRY TIMING ===
        rsi_value = rsi_15m_aligned[i]
        rsi_oversold = rsi_value < 40.0
        rsi_overbought = rsi_value > 60.0
        
        # === ENTRY LOGIC (LOOSE CONDITIONS) ===
        desired_signal = 0.0
        
        # LONG: 4h bull + 15m RSI oversold + session active
        # Volume filter is softer to ensure trades
        if htf_bull and rsi_oversold and session_active:
            desired_signal = SIZE
        # Weaker long: just HTF bull + RSI moderately oversold
        elif htf_bull and rsi_value < 45.0 and session_active:
            desired_signal = SIZE * 0.5
        
        # SHORT: 4h bear + 15m RSI overbought + session active
        elif htf_bear and rsi_overbought and session_active:
            desired_signal = -SIZE
        # Weaker short: just HTF bear + RSI moderately overbought
        elif htf_bear and rsi_value > 55.0 and session_active:
            desired_signal = -SIZE * 0.5
        
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
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
        elif abs(desired_signal) >= SIZE * 0.4:
            final_signal = np.sign(desired_signal) * SIZE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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