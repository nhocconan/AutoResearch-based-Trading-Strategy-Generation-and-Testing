#!/usr/bin/env python3
"""
Experiment #740: 1h Primary + 4h/12h HTF — HMA Trend + RSI Pullback + Volume Confluence

Hypothesis: After analyzing 495+ failed strategies, clear patterns emerge:
1. Complex regime (Chop+CRSI) = 0 trades (#727-735 all failed)
2. Session filters caused 0 trades on 1h (#735)
3. Simple HMA trend + RSI pullback on 4h got positive results (#739 template)
4. 1h entries within 4h/12h trend should give HTF quality with better timing

Strategy design:
1. 12h HMA(21) for macro trend bias (proven in best strategies)
2. 4h HMA(21) for intermediate trend confirmation
3. 1h RSI(14) pullback entries within HTF trend (loose: 35-65 range)
4. 1h price vs 4h HMA aligned for entry timing precision
5. Volume filter: volume > 0.7x 20-bar avg (loose to ensure trades)
6. ATR(14) trailing stop 2.5x for risk management
7. Discrete signals: 0.0, ±0.25 (conservative for 1h TF)

Key differences from failed experiments:
- NO session filter (caused 0 trades in #735)
- NO Choppiness Index (failed in 6+ experiments)
- LOOSE RSI filters (35-65, not extreme values)
- LOOSE volume filter (0.7x, not 1.5x)
- Clear hold logic to maintain positions through trends
- 1h timeframe with 4h/12h for direction (proven pattern)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 30-60 trades/year with strict confluence)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h12h_vol_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

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

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volume_ma(volume, period=20):
    """Volume moving average."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    sma_50_1h = calculate_sma(close, period=50)
    sma_200_1h = calculate_sma(close, period=200)
    vol_ma_1h = calculate_volume_ma(volume, period=20)
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Conservative for 1h timeframe
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):  # Need buffer for all indicators + HTF alignment
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(sma_50_1h[i]) or np.isnan(sma_200_1h[i]):
            continue
        if np.isnan(vol_ma_1h[i]) or vol_ma_1h[i] <= 1e-10:
            continue
        
        # === TREND BIAS (12h HTF HMA) - Macro direction ===
        trend_12h_bullish = close[i] > hma_12h_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # === TREND CONFIRMATION (4h HTF HMA) - Intermediate direction ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === VOLUME FILTER (loose to ensure trades) ===
        volume_ok = volume[i] > 0.7 * vol_ma_1h[i]
        
        # === RSI FILTERS (loose range to ensure trades) ===
        rsi_ok_long = 35 < rsi_1h[i] < 70  # Not overbought
        rsi_ok_short = 30 < rsi_1h[i] < 65  # Not oversold
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50_1h[i]
        above_sma200 = close[i] > sma_200_1h[i]
        below_sma50 = close[i] < sma_50_1h[i]
        below_sma200 = close[i] < sma_200_1h[i]
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        long_signal = False
        
        # Path 1: 12h bullish + 4h bullish + RSI pullback (40-55) + volume ok
        if trend_12h_bullish and trend_4h_bullish and 40 < rsi_1h[i] < 55 and volume_ok:
            long_signal = True
        
        # Path 2: 12h bullish + price > SMA50 + RSI ok + 4h bullish
        if trend_12h_bullish and above_sma50 and rsi_ok_long and trend_4h_bullish:
            long_signal = True
        
        # Path 3: Strong uptrend (above SMA50/200) + 12h bullish + RSI momentum
        if above_sma50 and above_sma200 and trend_12h_bullish and 45 < rsi_1h[i] < 65:
            long_signal = True
        
        # Path 4: 4h HMA cross above + 12h bullish + volume spike
        if trend_4h_bullish and trend_12h_bullish and volume[i] > 1.2 * vol_ma_1h[i]:
            long_signal = True
        
        if long_signal:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY CONDITIONS ===
        short_signal = False
        
        # Path 1: 12h bearish + 4h bearish + RSI rally (45-60) + volume ok
        if trend_12h_bearish and trend_4h_bearish and 45 < rsi_1h[i] < 60 and volume_ok:
            short_signal = True
        
        # Path 2: 12h bearish + price < SMA50 + RSI ok + 4h bearish
        if trend_12h_bearish and below_sma50 and rsi_ok_short and trend_4h_bearish:
            short_signal = True
        
        # Path 3: Strong downtrend (below SMA50/200) + 12h bearish + RSI momentum
        if below_sma50 and below_sma200 and trend_12h_bearish and 35 < rsi_1h[i] < 55:
            short_signal = True
        
        # Path 4: 4h HMA cross below + 12h bearish + volume spike
        if trend_4h_bearish and trend_12h_bearish and volume[i] > 1.2 * vol_ma_1h[i]:
            short_signal = True
        
        if short_signal:
            desired_signal = -BASE_SIZE
        
        # === CONFLICT RESOLUTION ===
        # If both long and short signals, go with 12h HMA trend (higher TF wins)
        if long_signal and short_signal:
            if trend_12h_bullish:
                desired_signal = BASE_SIZE
            elif trend_12h_bearish:
                desired_signal = -BASE_SIZE
            else:
                desired_signal = 0.0
        
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
                # Hold long if 12h HMA still bullish and 4h HMA still bullish
                if trend_12h_bullish and trend_4h_bullish:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 12h HMA still bearish and 4h HMA still bearish
                if trend_12h_bearish and trend_4h_bearish:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 12h trend reverses or 4h trend reverses
            if trend_12h_bearish or trend_4h_bearish:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 12h trend reverses or 4h trend reverses
            if trend_12h_bullish or trend_4h_bullish:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        
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
                # Position flip
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