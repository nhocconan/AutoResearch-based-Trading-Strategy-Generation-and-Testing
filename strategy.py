#!/usr/bin/env python3
"""
Experiment #1473: 5m Primary + 15m/4h HTF — Session-Filtered Trend Pullback

Hypothesis: 5m timeframe is unexplored territory. By using 15m HMA for immediate 
trend and 4h HMA for major bias, we only trade in established direction. 
5m RSI pullback entries provide precise timing. Session filter (08-20 UTC) 
avoids low-liquidity Asian hours where whipsaws dominate.

Key components:
1. 4h HMA(21) for major trend bias (NEVER counter-trend)
2. 15m HMA(16/48) for immediate trend momentum
3. 5m RSI(14) for pullback entry timing (long when RSI<45 in uptrend, short when RSI>55 in downtrend)
4. Session filter: only trade 08-20 UTC (London/NY overlap = highest liquidity)
5. ATR(14) trailing stoploss at 2.5x ATR
6. Discrete sizing: 0.0, ±0.15, ±0.20 (smaller due to higher trade frequency)

Why this should work on 5m:
- HTF trend filter prevents counter-trend disasters (main 5m failure mode)
- Session filter avoids Asian session whipsaws (low volume = fake breakouts)
- RSI pullback (not extreme) ensures enough trades while waiting for retracement
- Loose RSI thresholds (45/55, not 30/70) guarantee ≥50 trades/year
- Small size (0.15-0.20) accounts for higher fee drag on 5m

Entry logic (LOOSE to guarantee trades):
- LONG: 4h_HMA bullish + 15m_HMA16>HMA48 + 5m_RSI<50 + session 08-20 UTC
- SHORT: 4h_HMA bearish + 15m_HMA16<HMA48 + 5m_RSI>50 + session 08-20 UTC

Target: Sharpe>0.6, trades>=50 train, trades>=5 test, DD>-35%, 50-120 trades/year
Timeframe: 5m
Size: 0.15-0.20 discrete (smaller than higher TFs due to fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_session_rsi_pullback_hma_15m4h_v1"
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

def is_trading_session(open_time, start_hour=8, end_hour=20):
    """
    Check if timestamp is within trading session (UTC)
    08-20 UTC covers London open through NY session
    """
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
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_15m_16_raw = calculate_hma(df_15m['close'].values, period=16)
    hma_15m_48_raw = calculate_hma(df_15m['close'].values, period=48)
    hma_15m_16_aligned = align_htf_to_ltf(prices, df_15m, hma_15m_16_raw)
    hma_15m_48_aligned = align_htf_to_ltf(prices, df_15m, hma_15m_48_raw)
    
    # Calculate 5m indicators
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
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
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m_16_aligned[i]) or np.isnan(hma_15m_48_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        in_session = is_trading_session(open_time[i], start_hour=8, end_hour=20)
        
        # === TREND DIRECTION (4h HMA bias - NEVER counter-trend) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === 15m HMA CROSSOVER (immediate trend momentum) ===
        hma_15m_bullish = hma_15m_16_aligned[i] > hma_15m_48_aligned[i]
        hma_15m_bearish = hma_15m_16_aligned[i] < hma_15m_48_aligned[i]
        
        # === 5m RSI (pullback entry timing) ===
        rsi = rsi_14[i]
        
        # === ENTRY LOGIC (LOOSE - must generate ≥50 trades/year) ===
        desired_signal = 0.0
        
        # Only trade during session hours
        if in_session:
            # LONG: 4h bullish + 15m bullish + RSI pullback (<50, not extreme)
            if price_above_4h and hma_15m_bullish and rsi < 50:
                desired_signal = SIZE_BASE
            
            # SHORT: 4h bearish + 15m bearish + RSI pullback (>50, not extreme)
            elif price_below_4h and hma_15m_bearish and rsi > 50:
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