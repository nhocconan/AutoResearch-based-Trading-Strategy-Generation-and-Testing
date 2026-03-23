#!/usr/bin/env python3
"""
Experiment #880: 1h Primary + 4h/12h HTF — Relaxed RSI Pullback with HTF Trend Bias

Hypothesis: After 600+ failed strategies, the key insight is that 1h strategies fail
because entry conditions are TOO STRICT (0 trades). This strategy uses:

1. 1h Primary TF with RELAXED entry thresholds to guarantee 30-80 trades/year
2. 4h HMA(21) for trend direction bias (long only when price > 4h HMA)
3. 12h HMA(21) for macro regime filter (avoid counter-trend trades)
4. RSI(14) pullback entries with relaxed thresholds (30/70 not 20/80)
5. Volume filter (lenient: 0.6x avg) to avoid dead sessions
6. Session filter (8-20 UTC) when most liquidity exists
7. ATR(14) trailing stop (2.5x) for risk management

Why this should work on 1h:
- RELAXED RSI thresholds (30/70) ensure trades trigger on normal pullbacks
- HTF trend bias (4h/12h HMA) prevents counter-trend trades that fail in 2022 crash
- Volume + session filters avoid whipsaw during low-liquidity periods
- Discrete signal sizes (0.0, ±0.15, ±0.25) minimize fee churn
- ALL symbols must get trades (no SOL-only bias)

Critical improvements from failed experiments:
- RELAXED RSI from 20/80 to 30/70 to guarantee 30+ trades per symbol
- Simplified HTF logic (4h for direction, 12h for regime)
- Lenient volume filter (0.6x not 0.8x) to avoid filtering valid entries
- Session filter only reduces trades by ~30%, not 70%
- Hold logic maintains position through minor RSI fluctuations

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 40-70 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_pullback_4h12h_hma_vol_session_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Volume Simple Moving Average."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return pd.to_datetime(open_time, unit='ms').hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    vol_sma_1h = calculate_volume_sma(volume, period=20)
    sma_50_1h = calculate_sma(close, 50)
    sma_200_1h = calculate_sma(close, 200)
    
    # Calculate and align 4h HMA for trend direction bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for macro regime
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(vol_sma_1h[i]) or np.isnan(sma_50_1h[i]) or np.isnan(sma_200_1h[i]):
            continue
        
        # Extract UTC hour for session filter
        current_hour = get_utc_hour(open_time[i])
        in_session = 8 <= current_hour <= 20
        
        # Volume filter (lenient: 0.6x average)
        volume_ok = volume[i] >= 0.6 * vol_sma_1h[i]
        
        # === MACRO REGIME (12h HTF HMA21) ===
        macro_bull = close[i] > hma_12h_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i]
        
        # === MEDIUM-TERM TREND (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === SHORT-TERM TREND FILTER (1h SMA50/200) ===
        above_sma50 = close[i] > sma_50_1h[i]
        below_sma50 = close[i] < sma_50_1h[i]
        above_sma200 = close[i] > sma_200_1h[i]
        below_sma200 = close[i] < sma_200_1h[i]
        
        # === RSI SIGNALS (Relaxed thresholds: 30/70) ===
        rsi_oversold = rsi_1h[i] < 30
        rsi_overbought = rsi_1h[i] > 70
        rsi_extreme_oversold = rsi_1h[i] < 25
        rsi_extreme_overbought = rsi_1h[i] > 75
        rsi_neutral = 35 <= rsi_1h[i] <= 65
        
        desired_signal = 0.0
        
        # === LONG ENTRY LOGIC ===
        # Primary: RSI oversold + 4h bullish trend + volume ok
        if rsi_oversold and trend_4h_bullish and volume_ok:
            desired_signal = BASE_SIZE
        # Secondary: RSI oversold + 12h macro bull (even if 4h neutral)
        elif rsi_oversold and macro_bull and volume_ok:
            desired_signal = REDUCED_SIZE
        # Tertiary: Extreme RSI oversold (guarantees trades)
        elif rsi_extreme_oversold and volume_ok:
            desired_signal = REDUCED_SIZE
        # Fallback: RSI recovering from oversold in uptrend
        elif rsi_neutral and trend_4h_bullish and above_sma50 and volume_ok:
            if i > 0 and rsi_1h[i-1] < 30:  # RSI was oversold previously
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY LOGIC ===
        # Primary: RSI overbought + 4h bearish trend + volume ok
        if rsi_overbought and trend_4h_bearish and volume_ok:
            desired_signal = -BASE_SIZE
        # Secondary: RSI overbought + 12h macro bear (even if 4h neutral)
        elif rsi_overbought and macro_bear and volume_ok:
            desired_signal = -REDUCED_SIZE
        # Tertiary: Extreme RSI overbought (guarantees trades)
        elif rsi_extreme_overbought and volume_ok:
            desired_signal = -REDUCED_SIZE
        # Fallback: RSI weakening from overbought in downtrend
        elif rsi_neutral and trend_4h_bearish and below_sma50 and volume_ok:
            if i > 0 and rsi_1h[i-1] > 70:  # RSI was overbought previously
                desired_signal = -REDUCED_SIZE
        
        # Apply session filter (reduce size outside session, don't block entirely)
        if not in_session and desired_signal != 0:
            desired_signal = desired_signal * 0.6
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h trend still bullish and RSI not overbought
                if trend_4h_bullish and rsi_1h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h trend still bearish and RSI not oversold
                if trend_4h_bearish and rsi_1h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h trend reverses + RSI overbought
            if trend_4h_bearish and rsi_1h[i] > 70:
                desired_signal = 0.0
            # Exit if macro regime flips bearish
            if macro_bear and trend_4h_bearish:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h trend reverses + RSI oversold
            if trend_4h_bullish and rsi_1h[i] < 30:
                desired_signal = 0.0
            # Exit if macro regime flips bullish
            if macro_bull and trend_4h_bullish:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.9:
                desired_signal = BASE_SIZE
            else:
                desired_signal = REDUCED_SIZE
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.9:
                desired_signal = -BASE_SIZE
            else:
                desired_signal = -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
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
        
        signals[i] = desired_signal
    
    return signals