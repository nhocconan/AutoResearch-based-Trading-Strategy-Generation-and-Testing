#!/usr/bin/env python3
"""
Experiment #1493: 5m Primary + 15m/4h HTF — Session-Filtered Momentum Pullback

Hypothesis: 5m timeframe has ZERO prior experiments. This strategy uses:
1. 4h HMA(21) for major trend bias (NEVER trade counter-trend)
2. 15m RSI(14) for momentum confirmation (RSI>50 for longs, <50 for shorts)
3. 5m EMA(8/21) crossover for precise entry timing
4. Session filter: 12-22 UTC only (London/NY overlap = highest liquidity)
5. ATR(14) trailing stoploss (2.0x ATR)

Why this should work on 5m:
- HTF trend filter prevents 5m whipsaw deaths (proven in higher TFs)
- Session filter reduces noise during low-liquidity hours
- EMA cross on 5m = frequent enough signals (target 80-120 trades/year)
- LOOSE RSI threshold (45/55 not 30/70) ensures trades generate
- Small size (0.15) accounts for higher fee drag on lower TF

Entry logic (LOOSE to guarantee trades):
- LONG: 4h_HMA bullish + 15m_RSI>45 + 5m_EMA8>EMA21 + session active
- SHORT: 4h_HMA bearish + 15m_RSI<55 + 5m_EMA8<EMA21 + session active

Target: Sharpe>0.6, trades>=50 train, trades>=5 test, DD>-35%
Timeframe: 5m
Size: 0.15 discrete (smaller due to more trades)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_session_ema_pullback_4h15m_v1"
timeframe = "5m"
leverage = 1.0

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

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

def is_session_active(open_time):
    """
    Session filter: 12-22 UTC only (London/NY overlap + crypto peak hours)
    open_time is in milliseconds since epoch
    """
    # Convert to hour of day UTC
    hour = (open_time // (1000 * 60 * 60)) % 24
    return 12 <= hour <= 22

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_15m = get_htf_data(prices, '15m')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    rsi_15m_raw = calculate_rsi(df_15m['close'].values, period=14)
    rsi_15m_aligned = align_htf_to_ltf(prices, df_15m, rsi_15m_raw)
    
    # Calculate 5m indicators
    ema_8 = calculate_ema(close, period=8)
    ema_21 = calculate_ema(close, period=21)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.15  # Smaller size for 5m due to higher fee drag
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(ema_8[i]) or np.isnan(ema_21[i]):
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
        
        # === SESSION FILTER (12-22 UTC only) ===
        session_active = is_session_active(open_time[i])
        
        # === TREND DIRECTION (4h HMA bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === MOMENTUM (15m RSI) ===
        rsi_15m = rsi_15m_aligned[i]
        momentum_bullish = rsi_15m > 45  # LOOSE threshold
        momentum_bearish = rsi_15m < 55  # LOOSE threshold
        
        # === ENTRY SIGNAL (5m EMA crossover) ===
        ema_bullish = ema_8[i] > ema_21[i]
        ema_bearish = ema_8[i] < ema_21[i]
        
        # Check for fresh crossover (not just sustained)
        ema_cross_long = ema_bullish and (i > 0 and ema_8[i-1] <= ema_21[i-1] if not np.isnan(ema_8[i-1]) else False)
        ema_cross_short = ema_bearish and (i > 0 and ema_8[i-1] >= ema_21[i-1] if not np.isnan(ema_8[i-1]) else False)
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        if session_active:
            # LONG: 4h bullish + 15m RSI not bearish + 5m EMA cross or sustained bullish
            if price_above_4h and momentum_bullish and ema_bullish:
                desired_signal = SIZE
            
            # SHORT: 4h bearish + 15m RSI not bullish + 5m EMA cross or sustained bearish
            elif price_below_4h and momentum_bearish and ema_bearish:
                desired_signal = -SIZE
        
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
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
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