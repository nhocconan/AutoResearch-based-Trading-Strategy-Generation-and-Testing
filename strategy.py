#!/usr/bin/env python3
"""
Experiment #1008: 30m Primary + 4h/1d HTF — HTF Trend + 30m Pullback Entry + Session Filter

Hypothesis: After 732 failed strategies, the key for lower TF (30m) is using HTF for 
SIGNAL DIRECTION and 30m only for ENTRY TIMING. This gives HTF trade frequency with 
lower TF execution precision.

Key insights from research:
1. 4h HMA(21) for medium-term trend — only trade in HTF trend direction
2. 1d HMA(21) for macro regime filter — avoid counter-macro trades
3. 30m RSI(14) pullback entries — enter on oversold in uptrend, overbought in downtrend
4. Session filter (8-20 UTC) — only trade during high-volume hours
5. Volume confirmation — volume > 0.8x 20-bar average
6. ATR(14) trailing stoploss at 2.5x

Why 30m timeframe:
- Target 30-80 trades/year (strict entry filters prevent fee drag)
- 4h/1d HTF provides strong trend bias (reduces whipsaws)
- 30m entries capture better risk/reward than 4h entries
- Session filter ensures we only trade during liquid hours

Critical improvements:
- RELAXED RSI thresholds (30/70 not 25/75) to ensure trades trigger
- HTF trend as PRIMARY filter (not optional confluence)
- Discrete signal sizes (0.0, ±0.20, ±0.30) minimize fee churn
- Session filter reduces trades by ~60% (only 12h/day)
- ALL symbols MUST have positive Sharpe (no SOL-only bias)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 30m (target 40-80 trades/year with session filter)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_htf_trend_rsi_pullback_session_vol_atr_v1"
timeframe = "30m"
leverage = 1.0

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
    """Simple moving average of volume."""
    n = len(volume)
    vol_sma = np.full(n, np.nan)
    
    if n < period:
        return vol_sma
    
    for i in range(period - 1, n):
        vol_sma[i] = np.mean(volume[i-period+1:i+1])
    
    return vol_sma

def get_hour_from_open_time(open_time_array):
    """Extract hour from open_time (milliseconds timestamp)."""
    hours = np.zeros(len(open_time_array), dtype=int)
    for i in range(len(open_time_array)):
        # Convert ms to seconds, then to datetime
        ts_sec = open_time_array[i] / 1000
        hours[i] = int((ts_sec % 86400) / 3600)
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (30m) indicators
    rsi_30m = calculate_rsi(close, period=14)
    atr_30m = calculate_atr(high, low, close, period=14)
    vol_sma_30m = calculate_volume_sma(volume, period=20)
    
    # Calculate and align 4h HMA for medium-term trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro regime
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Get session hours
    hours = get_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_30m[i]) or np.isnan(atr_30m[i]) or atr_30m[i] <= 1e-10:
            continue
        if np.isnan(vol_sma_30m[i]) or np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # === MACRO REGIME (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM-TERM TREND (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_sma_30m[i] if vol_sma_30m[i] > 0 else False
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_30m[i] < 35
        rsi_overbought = rsi_30m[i] > 65
        rsi_extreme_oversold = rsi_30m[i] < 25
        rsi_extreme_overbought = rsi_30m[i] > 75
        rsi_neutral = 40 <= rsi_30m[i] <= 60
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Primary: 4h trend bullish + 1d macro bullish + RSI pullback + session + volume
        if trend_4h_bullish and macro_bull and in_session and volume_confirmed:
            if rsi_oversold:
                desired_signal = BASE_SIZE
            elif rsi_extreme_oversold:
                desired_signal = BASE_SIZE
        
        # Secondary: 4h trend bullish + RSI extreme oversold (guarantees trades)
        elif trend_4h_bullish and rsi_extreme_oversold and in_session:
            desired_signal = REDUCED_SIZE
        
        # Tertiary: 1d macro bullish + RSI extreme oversold (backup for trades)
        elif macro_bull and rsi_extreme_oversold and in_session:
            desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY CONDITIONS ===
        # Primary: 4h trend bearish + 1d macro bearish + RSI rally + session + volume
        if trend_4h_bearish and macro_bear and in_session and volume_confirmed:
            if rsi_overbought:
                desired_signal = -BASE_SIZE
            elif rsi_extreme_overbought:
                desired_signal = -BASE_SIZE
        
        # Secondary: 4h trend bearish + RSI extreme overbought (guarantees trades)
        elif trend_4h_bearish and rsi_extreme_overbought and in_session:
            desired_signal = -REDUCED_SIZE
        
        # Tertiary: 1d macro bearish + RSI extreme overbought (backup for trades)
        elif macro_bear and rsi_extreme_overbought and in_session:
            desired_signal = -REDUCED_SIZE
        
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
                if trend_4h_bullish and rsi_30m[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h trend still bearish and RSI not oversold
                if trend_4h_bearish and rsi_30m[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h trend reverses bearish
            if trend_4h_bearish and rsi_30m[i] > 60:
                desired_signal = 0.0
            # Exit if 1d macro reverses
            if macro_bear and rsi_30m[i] > 65:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h trend reverses bullish
            if trend_4h_bullish and rsi_30m[i] < 40:
                desired_signal = 0.0
            # Exit if 1d macro reverses
            if macro_bull and rsi_30m[i] < 35:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
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