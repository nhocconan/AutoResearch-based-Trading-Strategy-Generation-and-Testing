#!/usr/bin/env python3
"""
Experiment #021: 15m Primary + 1h/4h HTF — RSI Mean Reversion + HMA Trend Bias

Hypothesis: 15m strategies failed because entry conditions were TOO STRICT (0 trades).
Solution: Simplify to RSI(7) mean-reversion with soft HTF bias (not hard filter).
- 15m RSI(7) oversold/overbought triggers entries (fast enough to generate trades)
- 4h HMA(21) provides trend bias (longs preferred when price > 4h HMA, shorts when <)
- 1h ATR for stoploss (2x ATR trailing)
- Session filter: UTC 00-12 only (London/NY overlap, higher volume)
- Position size: 0.18 (smaller for 15m frequency, target 50-100 trades/year)
- Exit: RSI crosses 50 OR trailing stop hit

Why this should work on 15m:
- RSI(7) is fast enough to trigger 2-3 signals per week per symbol
- HTF bias is SOFT (increases size, doesn't block entries)
- Mean-reversion works well on intraday 15m timeframe
- Session filter reduces noise during low-volume hours
- Conservative size (0.18) protects against 2022-style crashes

Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_hma_session_4h1h_v1"
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

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
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
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1h ATR for stoploss
    atr_1h_raw = calculate_atr(df_1h['high'].values, df_1h['low'].values, df_1h['close'].values, period=14)
    atr_1h_aligned = align_htf_to_ltf(prices, df_1h, atr_1h_raw)
    
    # Calculate primary (15m) indicators
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr_15m = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.18  # 18% position size (conservative for 15m)
    SIZE_TREND = 0.25  # 25% when trend aligns
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_15m[i]) or atr_15m[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(atr_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (UTC 00-12 only) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        in_session = (hour_utc >= 0 and hour_utc < 12)
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === PRIMARY SIGNAL (15m RSI Mean Reversion) ===
        # LONG: RSI(7) < 25 (oversold)
        # SHORT: RSI(7) > 75 (overbought)
        rsi_oversold = rsi_7[i] < 25.0
        rsi_overbought = rsi_7[i] > 75.0
        
        # RSI recovery confirmation (for exit timing)
        rsi_recovering_long = rsi_7[i] > 35.0 and rsi_7[i-1] <= 35.0
        rsi_recovering_short = rsi_7[i] < 65.0 and rsi_7[i-1] >= 65.0
        
        # === SMA200 FILTER (soft, not hard) ===
        # Prefer longs when price > SMA200, shorts when price < SMA200
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        use_trend_size = False
        
        if in_session:
            # LONG entries
            if rsi_oversold:
                if htf_bull and above_sma200:
                    # All aligned: trend + HTF + SMA
                    desired_signal = SIZE_TREND
                    use_trend_size = True
                elif htf_bull or above_sma200:
                    # Partial alignment
                    desired_signal = SIZE_BASE
                else:
                    # Counter-trend but RSI extreme
                    desired_signal = SIZE_BASE * 0.6
            
            # SHORT entries
            elif rsi_overbought:
                if htf_bear and below_sma200:
                    # All aligned: trend + HTF + SMA
                    desired_signal = -SIZE_TREND
                    use_trend_size = True
                elif htf_bear or below_sma200:
                    # Partial alignment
                    desired_signal = -SIZE_BASE
                else:
                    # Counter-trend but RSI extreme
                    desired_signal = -SIZE_BASE * 0.6
        
        # === EXIT SIGNALS (RSI recovery) ===
        if in_position and position_side > 0 and rsi_recovering_long:
            desired_signal = 0.0  # Take profit on long
        
        if in_position and position_side < 0 and rsi_recovering_short:
            desired_signal = 0.0  # Take profit on short
        
        # === STOPLOSS CHECK (Trailing ATR 2x from 1h) ===
        stoploss_triggered = False
        use_1h_atr = atr_1h_aligned[i] if not np.isnan(atr_1h_aligned[i]) else atr_15m[i]
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * use_1h_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * use_1h_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_TREND * 0.9:
            final_signal = SIZE_TREND
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.5
        elif desired_signal <= -SIZE_TREND * 0.9:
            final_signal = -SIZE_TREND
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = use_1h_atr
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = use_1h_atr
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals