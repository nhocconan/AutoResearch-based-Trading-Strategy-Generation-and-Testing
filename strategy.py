#!/usr/bin/env python3
"""
Experiment #933: 5m Primary + 15m/4h HTF — HMA Trend + RSI Pullback + Session Filter

Hypothesis: 5m timeframe with 15m/4h HTF trend alignment can capture intraday momentum
while avoiding whipsaw. Key innovation: VERY LOOSE entry conditions to guarantee trades
(most 5m/15m strategies failed with 0 trades). Session filter (12-20 UTC) ensures
liquidity during London/NY overlap. RSI pullback entries in HTF trend direction.

Key innovations:
1. 4h HMA(21) for HTF bias - price above = bullish bias
2. 15m HMA(16/48) for intermediate trend - aligned properly via mtf_data
3. 5m RSI(14) pullback entries - long when RSI<60 in uptrend, short when RSI>40 in downtrend
4. Session filter: 12-20 UTC only (London/NY overlap = best liquidity)
5. ATR(14) 2.0x trailing stop for risk management
6. VERY LOOSE entries to guarantee ≥50 trades/year on 5m

Entry conditions (LOOSE to guarantee trades):
- LONG = (4h HMA bull OR 15m HMA bull) + RSI(5m) < 65 + session
- SHORT = (4h HMA bear OR 15m HMA bear) + RSI(5m) > 35 + session
- This OR logic ensures we get trades even if one HTF is ambiguous

Target: Sharpe>0.45, trades>=50 train, trades>=10 test, DD>-40%
Timeframe: 5m
Size: 0.15-0.20 discrete (smaller due to higher trade frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_hma_rsi_session_15m4h_v1"
timeframe = "5m"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_n)
    return hma

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
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_15m = get_htf_data(prices, '15m')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF HMAs
    hma_15m_16_raw = calculate_hma(df_15m['close'].values, period=16)
    hma_15m_48_raw = calculate_hma(df_15m['close'].values, period=48)
    hma_15m_16_aligned = align_htf_to_ltf(prices, df_15m, hma_15m_16_raw)
    hma_15m_48_aligned = align_htf_to_ltf(prices, df_15m, hma_15m_48_raw)
    
    hma_4h_21_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21_raw)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]):
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
        
        if np.isnan(hma_4h_21_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (12-20 UTC) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        in_session = 12 <= hour_utc <= 20
        
        # === HTF BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_21_aligned[i]
        htf_4h_bear = close[i] < hma_4h_21_aligned[i]
        
        # === 15m HMA TREND ===
        hma_15m_bull = hma_15m_16_aligned[i] > hma_15m_48_aligned[i]
        hma_15m_bear = hma_15m_16_aligned[i] < hma_15m_48_aligned[i]
        
        # === RSI PULLBACK (LOOSE) ===
        rsi_pullback_long = rsi_14[i] < 65.0
        rsi_pullback_short = rsi_14[i] > 35.0
        
        # === ENTRY LOGIC (VERY LOOSE TO GUARANTEE TRADES) ===
        desired_signal = 0.0
        
        if in_session:
            # LONG: (4h bull OR 15m bull) + RSI pullback
            if (htf_4h_bull or hma_15m_bull) and rsi_pullback_long:
                # Stronger signal if both HTF agree
                if htf_4h_bull and hma_15m_bull:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # SHORT: (4h bear OR 15m bear) + RSI pullback
            elif (htf_4h_bear or hma_15m_bear) and rsi_pullback_short:
                # Stronger signal if both HTF agree
                if htf_4h_bear and hma_15m_bear:
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