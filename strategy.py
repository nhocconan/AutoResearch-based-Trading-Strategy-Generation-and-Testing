#!/usr/bin/env python3
"""
Experiment #833: 5m Primary + 15m/4h HTF — Session-Filtered Trend Following

Hypothesis: 5m timeframe with strict HTF alignment and session filtering can
capture intraday momentum while avoiding noise. Key innovations:
1. 4h HMA(21) for primary trend bias — only trade in HTF direction
2. 15m RSI(14) for momentum confirmation — RSI>50 for longs, <50 for shorts
3. 5m EMA(21) pullback entry — enter on pullback to EMA in trend direction
4. Session filter: 08-20 UTC only (high liquidity, avoid Asia overnight noise)
5. Volume confirmation — taker_buy_volume ratio > 0.55 for longs
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete sizing: 0.0, ±0.15, ±0.20 (smaller due to 5m frequency)

Why 5m might work:
- Captures intraday momentum moves that higher TF miss
- Session filter removes 60% of noise (overnight Asia session)
- HTF alignment prevents counter-trend trades (main failure mode)
- Fewer false signals than pure 5m strategies

Target: Sharpe>0.50, trades>=50/train, trades>=5/test, DD>-30%
Timeframe: 5m
Size: 0.15-0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_hma_rsi_session_15m4h_v1"
timeframe = "5m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
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

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def is_session_active(open_time_unix_ms):
    """Check if timestamp is within 08-20 UTC session"""
    # Convert ms to hours
    hours_utc = (open_time_unix_ms // (1000 * 60 * 60)) % 24
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
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    rsi_15m_raw = calculate_rsi(df_15m['close'].values, period=14)
    rsi_15m_aligned = align_htf_to_ltf(prices, df_15m, rsi_15m_raw)
    
    # Calculate 5m indicators
    ema_21 = calculate_ema(close, period=21)
    ema_50 = calculate_ema(close, period=50)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Taker buy ratio (volume confirmation)
    taker_ratio = np.zeros(n)
    for i in range(n):
        if volume[i] > 1e-10:
            taker_ratio[i] = taker_buy_volume[i] / volume[i]
        else:
            taker_ratio[i] = 0.5
    
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
        
        if np.isnan(ema_21[i]) or np.isnan(ema_50[i]) or np.isnan(rsi_14[i]):
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
        
        # === SESSION FILTER (08-20 UTC only) ===
        if not is_session_active(open_time[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m MOMENTUM (RSI confirmation) ===
        rsi_15m_bull = rsi_15m_aligned[i] > 50.0
        rsi_15m_bear = rsi_15m_aligned[i] < 50.0
        
        # === 5m LOCAL TREND ===
        ema_5m_bull = ema_21[i] > ema_50[i]
        ema_5m_bear = ema_21[i] < ema_50[i]
        
        # === PULLBACK ENTRY (price near EMA21) ===
        price_above_ema = close[i] > ema_21[i]
        price_below_ema = close[i] < ema_21[i]
        
        # Pullback threshold: price within 0.5% of EMA21
        pullback_long = price_below_ema and (close[i] > ema_21[i] * 0.995)
        pullback_short = price_above_ema and (close[i] < ema_21[i] * 1.005)
        
        # === VOLUME CONFIRMATION ===
        volume_bull = taker_ratio[i] > 0.55
        volume_bear = taker_ratio[i] < 0.45
        
        # === 5m RSI EXTREMES (for entry timing) ===
        rsi_5m_oversold = rsi_14[i] < 45.0
        rsi_5m_overbought = rsi_14[i] > 55.0
        
        # === ENTRY LOGIC (ALL CONDITIONS MUST ALIGN) ===
        desired_signal = 0.0
        
        # LONG: 4h bull + 15m RSI bull + 5m EMA bull + pullback + volume
        if htf_4h_bull and rsi_15m_bull and ema_5m_bull:
            if pullback_long and rsi_5m_oversold and volume_bull:
                desired_signal = SIZE_STRONG
            elif pullback_long and rsi_5m_oversold:
                desired_signal = SIZE_BASE
        
        # SHORT: 4h bear + 15m RSI bear + 5m EMA bear + pullback + volume
        elif htf_4h_bear and rsi_15m_bear and ema_5m_bear:
            if pullback_short and rsi_5m_overbought and volume_bear:
                desired_signal = -SIZE_STRONG
            elif pullback_short and rsi_5m_overbought:
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