#!/usr/bin/env python3
"""
Experiment #489: 15m Primary + 1h/1d HTF — RSI Mean Reversion with Trend Filter

Hypothesis: Previous 15m experiments failed with 0 trades due to TOO STRICT entry conditions.
This strategy uses LOOSE RSI entries with HTF trend alignment to guarantee trade generation:
1. 1d HMA(21) = major trend bias (long only above, short only below)
2. 1h HMA(21) = intermediate trend confirmation
3. 15m RSI(7) = fast entry trigger (crosses back from extreme, not just at extreme)
4. Session filter: 00-12 UTC only (London/NY overlap, reduces trade count naturally)
5. ATR(14)*2.0 stoploss on all positions

Key differences from failed 15m experiments:
- RSI(7) not RSI(14) — faster, more signals on 15m
- Entry on RSI CROSSBACK (e.g., was <20, now >20) not just at extreme
- ONLY 2 HTF filters (1d + 1h), not 3+ confluence
- Session filter reduces trades naturally without killing signal generation
- Size = 0.18 (smaller for 15m frequency, target 50-80 trades/year)

Target: Sharpe>0.40, trades>=150 train (37/year), trades>=20 test
Timeframe: 15m (FIRST 15m experiment with actual trade generation)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_crosstab_1h1d_session_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMAs
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    # Calculate 15m indicators
    hma_15m = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi_fast = calculate_rsi(close, period=7)   # Fast RSI for 15m entries
    rsi_slow = calculate_rsi(close, period=14)  # Slow RSI for confirmation
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.18      # Base position size for 15m (smaller due to frequency)
    SIZE_STRONG = 0.22    # Stronger conviction entries
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m[i]) or np.isnan(rsi_fast[i]) or np.isnan(rsi_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER: Only trade 00-12 UTC (London/NY overlap) ===
        # open_time is in milliseconds since epoch
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        in_session = (hour_utc >= 0 and hour_utc < 12)
        
        # === 1d HTF MAJOR TREND BIAS ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 1h HTF INTERMEDIATE TREND ===
        htf_1h_bull = close[i] > hma_1h_aligned[i]
        htf_1h_bear = close[i] < hma_1h_aligned[i]
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === RSI EXTREMES AND CROSSBACK ===
        # Long: RSI was <20, now >20 (crossing back from oversold)
        rsi_oversold_crossback = (rsi_fast[i] > 20.0 and rsi_fast[i-1] <= 20.0)
        rsi_extreme_oversold = rsi_fast[i] < 25.0
        
        # Short: RSI was >80, now <80 (crossing back from overbought)
        rsi_overbought_crossback = (rsi_fast[i] < 80.0 and rsi_fast[i-1] >= 80.0)
        rsi_extreme_overbought = rsi_fast[i] > 75.0
        
        # === ENTRY LOGIC (LOOSE - designed to generate trades) ===
        desired_signal = 0.0
        
        # LONG ENTRIES (must have 1d bullish bias)
        if htf_bull and in_session:
            # Strong long: 1h also bull + RSI crossback from oversold
            if htf_1h_bull and rsi_oversold_crossback:
                desired_signal = SIZE_STRONG
            # Standard long: RSI crossback + above SMA50
            elif rsi_oversold_crossback and above_sma50:
                desired_signal = SIZE_BASE
            # Mean reversion long: RSI extreme + above SMA200 (no crossback needed)
            elif rsi_extreme_oversold and above_sma200:
                desired_signal = SIZE_BASE * 0.8
        
        # SHORT ENTRIES (must have 1d bearish bias)
        elif htf_bear and in_session:
            # Strong short: 1h also bear + RSI crossback from overbought
            if htf_1h_bear and rsi_overbought_crossback:
                desired_signal = -SIZE_STRONG
            # Standard short: RSI crossback + below SMA50
            elif rsi_overbought_crossback and below_sma50:
                desired_signal = -SIZE_BASE
            # Mean reversion short: RSI extreme + below SMA200
            elif rsi_extreme_overbought and below_sma200:
                desired_signal = -SIZE_BASE * 0.8
        
        # === STOPLOSS CHECK (2.0x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
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
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.8
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
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
        
        signals[i] = final_signal
    
    return signals