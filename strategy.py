#!/usr/bin/env python3
"""
Experiment #736: 12h Primary + 1d HTF — Donchian Breakout with HMA Trend Filter

Hypothesis: After 494 failed strategies, the pattern is clear — complex regime detection
(Choppiness + CRSI + multiple filters) causes 0 trades. This strategy uses SIMPLE logic:
1. 1d HMA(21) for trend bias (proven in best strategies)
2. 12h Donchian(20) breakout for entries (simple, generates trades consistently)
3. 12h RSI(14) for timing filter (loose: 40/60 thresholds)
4. ATR(14) trailing stop 2.5x for risk management
5. Discrete signal sizes: 0.0, ±0.25, ±0.30

Key differences from failed #732:
- Removed Choppiness Index (causes 0 trades when threshold too strict)
- Removed CRSI complexity (failed in 6+ experiments)
- Simpler Donchian breakout logic (price breaks 20-bar high/low)
- Looser RSI filters to ensure trade frequency
- Clear hold logic to maintain positions through trend

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_1d_atr_v1"
timeframe = "12h"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel for breakout detection."""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (12h) indicators
    rsi_12h = calculate_rsi(close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    sma_50 = calculate_sma(close, period=50)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
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
    
    for i in range(250, n):  # Need buffer for all indicators + HTF alignment
        # Skip if indicators not ready
        if np.isnan(rsi_12h[i]) or np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            continue
        
        # === TREND BIAS (1d HTF HMA) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma50 = close[i] < sma_50[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI FILTERS (loose to ensure trades) ===
        rsi_neutral_long = rsi_12h[i] < 65  # Not overbought
        rsi_neutral_short = rsi_12h[i] > 35  # Not oversold
        rsi_momentum_long = rsi_12h[i] > 45  # Some momentum
        rsi_momentum_short = rsi_12h[i] < 55  # Some downward momentum
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        long_signal = False
        
        # Path 1: Donchian breakout + bullish 1d trend + RSI momentum
        if close[i] > donch_upper[i-1] and trend_1d_bullish and rsi_momentum_long:
            long_signal = True
        
        # Path 2: Price above SMA50 + bullish 1d trend + RSI not overbought
        if above_sma50 and trend_1d_bullish and rsi_neutral_long and rsi_12h[i] > 40:
            long_signal = True
        
        # Path 3: Price above SMA200 + pullback to Donchian mid + bullish trend
        donch_mid = (donch_upper[i] + donch_lower[i]) / 2
        if above_sma200 and trend_1d_bullish and close[i] < donch_mid and rsi_12h[i] > 35:
            long_signal = True
        
        # Path 4: Strong trend (above both SMA50/200) + 1d bullish
        if above_sma50 and above_sma200 and trend_1d_bullish and rsi_12h[i] > 40:
            long_signal = True
        
        if long_signal:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY CONDITIONS ===
        short_signal = False
        
        # Path 1: Donchian breakdown + bearish 1d trend + RSI momentum
        if close[i] < donch_lower[i-1] and trend_1d_bearish and rsi_momentum_short:
            short_signal = True
        
        # Path 2: Price below SMA50 + bearish 1d trend + RSI not oversold
        if below_sma50 and trend_1d_bearish and rsi_neutral_short and rsi_12h[i] < 60:
            short_signal = True
        
        # Path 3: Price below SMA200 + rally to Donchian mid + bearish trend
        if below_sma200 and trend_1d_bearish and close[i] > donch_mid and rsi_12h[i] < 65:
            short_signal = True
        
        # Path 4: Strong downtrend (below both SMA50/200) + 1d bearish
        if below_sma50 and below_sma200 and trend_1d_bearish and rsi_12h[i] < 60:
            short_signal = True
        
        if short_signal:
            desired_signal = -BASE_SIZE
        
        # === CONFLICT RESOLUTION ===
        # If both long and short signals, go with 1d HMA trend
        if long_signal and short_signal:
            if trend_1d_bullish:
                desired_signal = BASE_SIZE
            elif trend_1d_bearish:
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
                # Hold long if 1d HMA still bullish and RSI not extremely overbought
                if trend_1d_bullish and rsi_12h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 1d HMA still bearish and RSI not extremely oversold
                if trend_1d_bearish and rsi_12h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 1d trend reverses or RSI extremely overbought
            if trend_1d_bearish or rsi_12h[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1d trend reverses or RSI extremely oversold
            if trend_1d_bullish or rsi_12h[i] < 20:
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
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
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