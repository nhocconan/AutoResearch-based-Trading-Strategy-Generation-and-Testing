#!/usr/bin/env python3
"""
Experiment #269: 15m Primary + 1h/1d HTF — Trend-Filtered Mean Reversion v1

Hypothesis: 15m timeframe with strict HTF trend filter can capture intraday mean-reversion
while avoiding counter-trend trades that destroy returns. Key design:

1. HTF DIRECTION LOCK: Only trade long when 1d HMA(50) bullish, short when bearish
2. INTERMEDIATE CONFIRMATION: 1h HMA(21) must agree with 1d direction
3. 15m ENTRY: RSI(7) extremes ( <25 long, >75 short) + price vs EMA(21) pullback
4. SESSION FILTER: Only trade 00-12 UTC (London+NY overlap = higher volume)
5. VOLATILITY FILTER: ATR(14) > median ATR (avoid dead/choppy markets)
6. STOPLOSS: 2.0x ATR trailing stop

Why this might work on 15m:
- HTF filters reduce trade frequency to 40-100/year (critical for fee drag)
- Mean reversion on 15m works better than trend-following (less whipsaw)
- Session filter avoids low-volume Asian session chop
- Smaller position size (0.15-0.25) accounts for higher frequency

Position sizing: 0.15 base, 0.25 strong (discrete levels to minimize churn)
Target: Sharpe>0.40, DD>-40%, trades>=30 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_trend_filtered_mr_rsi_session_1h1d_v1"
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

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def get_hour_from_open_time(prices):
    """Extract UTC hour from open_time column"""
    # open_time is in milliseconds since epoch
    open_time_ms = prices["open_time"].values
    hours = (open_time_ms // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Extract UTC hour for session filter
    hours = get_hour_from_open_time(prices)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    # Calculate primary (15m) indicators
    ema_15m = calculate_ema(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=7)  # Faster RSI for 15m
    sma_200 = calculate_sma(close, 200)
    
    # Calculate median ATR for volatility filter (use last 500 bars)
    atr_median = np.nanmedian(atr[-500:]) if n >= 500 else np.nanmedian(atr[100:])
    if np.isnan(atr_median) or atr_median < 1e-10:
        atr_median = np.nanpercentile(atr[~np.isnan(atr)], 50) if np.any(~np.isnan(atr)) else 1.0
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_15m[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1h_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_200[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC = London+NY overlap) ===
        in_session = hours[i] < 12  # 00:00 to 11:59 UTC
        
        # === VOLATILITY FILTER (avoid dead markets) ===
        vol_ok = atr[i] > atr_median * 0.7  # At least 70% of median ATR
        
        # === HTF TREND BIAS ===
        # 1d HMA for major trend direction
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # 1h HMA for intermediate confirmation
        htf_1h_bull = close[i] > hma_1h_aligned[i]
        htf_1h_bear = close[i] < hma_1h_aligned[i]
        
        # === 15m EMA TREND ===
        ema_bull = close[i] > ema_15m[i]
        ema_bear = close[i] < ema_15m[i]
        
        # === SMA200 FILTER (major trend) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI EXTREMES (mean reversion entries) ===
        rsi_oversold = rsi[i] < 25.0  # Long entry
        rsi_overbought = rsi[i] > 75.0  # Short entry
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: 1d bullish + 1h bullish + RSI oversold + in session + vol ok
        # Entry on pullback (price < EMA) or bounce (price > EMA but RSI extreme)
        if htf_1d_bull and htf_1h_bull and above_sma200:
            if rsi_oversold and in_session and vol_ok:
                # Strong signal if price also below EMA (pullback)
                if ema_bear:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT: 1d bearish + 1h bearish + RSI overbought + in session + vol ok
        elif htf_1d_bear and htf_1h_bear and below_sma200:
            if rsi_overbought and in_session and vol_ok:
                # Strong signal if price also above EMA (pullback)
                if ema_bull:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
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
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i] if position_side > 0 else 0.0
                lowest_since_entry = low[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i] if position_side > 0 else 0.0
                lowest_since_entry = low[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
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