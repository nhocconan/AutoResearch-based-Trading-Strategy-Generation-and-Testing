#!/usr/bin/env python3
"""
Experiment #621: 15m Primary + 1h/4h/1d HTF — RSI Mean Reversion + Trend Filter

Hypothesis: 15m timeframe is completely unexplored (0 successful experiments). Previous 15m 
strategies failed because entry conditions were TOO STRICT (0 trades). This strategy uses:
1. SIMPLE entry logic: 4h HMA for direction + 15m RSI(7) extremes for timing
2. LOOSE filters: RSI<35 for long, RSI>65 for short (not extreme 20/80)
3. Session bias: prefer 00-12 UTC (London/NY overlap) but allow all hours
4. ATR stoploss: 2.0x ATR trailing stop
5. Target 60-100 trades/year (not too few, not too many)

Key insight from failures: #617 15m had 0 trades because filters were too strict.
This strategy uses MINIMAL confluence: just HTF trend + RSI extreme.

Strategy logic:
1. 4h HMA(21) = trend direction (bull if price>HMA, bear if price<HMA)
2. 15m RSI(7) = entry timing (oversold<35, overbought>65)
3. 15m ATR(14) = stoploss (2.0x ATR trailing)
4. Session filter: boost size during 00-12 UTC, reduce otherwise
5. Discrete signals: 0.0, ±0.20, ±0.25

Target: Sharpe>0.40, trades>=40 train (10/year), trades>=5 test
Timeframe: 15m
Size: 0.20-0.25 discrete (smaller for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_meanrev_hma_4h_session_v1"
timeframe = "15m"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
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
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 1D MACRO BIAS (optional filter) ===
        macro_bull = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        macro_bear = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # === 15m RSI SIGNALS (LOOSE thresholds for more trades) ===
        rsi_oversold = rsi_7[i] < 35.0
        rsi_overbought = rsi_7[i] > 65.0
        rsi_extreme_oversold = rsi_7[i] < 25.0
        rsi_extreme_overbought = rsi_7[i] > 75.0
        
        # RSI crossing up from oversold
        rsi_cross_up = False
        rsi_cross_down = False
        if i > 0 and not np.isnan(rsi_7[i-1]):
            rsi_cross_up = rsi_7[i] > 35.0 and rsi_7[i-1] <= 35.0
            rsi_cross_down = rsi_7[i] < 65.0 and rsi_7[i-1] >= 65.0
        
        # === SESSION FILTER (00-12 UTC = London/NY overlap) ===
        # Parse hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        is_peak_session = 0 <= hour_utc <= 12  # London + NY overlap
        
        # === SMA200 FILTER (long-term trend) ===
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === ENTRY LOGIC (SIMPLE - maximize trade count) ===
        desired_signal = 0.0
        
        # LONG entries: HTF bull + RSI oversold (mean reversion in uptrend)
        if htf_bull:
            # Primary: RSI extreme oversold in bull trend
            if rsi_extreme_oversold:
                desired_signal = SIZE_STRONG if is_peak_session else SIZE_BASE
            # Secondary: RSI crosses up from oversold
            elif rsi_cross_up:
                desired_signal = SIZE_BASE
            # Tertiary: RSI moderately oversold in strong bull (above SMA200)
            elif rsi_oversold and above_sma200:
                desired_signal = SIZE_BASE * 0.8
        
        # SHORT entries: HTF bear + RSI overbought (mean reversion in downtrend)
        elif htf_bear:
            # Primary: RSI extreme overbought in bear trend
            if rsi_extreme_overbought:
                desired_signal = -SIZE_STRONG if is_peak_session else -SIZE_BASE
            # Secondary: RSI crosses down from overbought
            elif rsi_cross_down:
                desired_signal = -SIZE_BASE
            # Tertiary: RSI moderately overbought in strong bear (below SMA200)
            elif rsi_overbought and below_sma200:
                desired_signal = -SIZE_BASE * 0.8
        
        # === STOPLOSS CHECK (2.0x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
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
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.8
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