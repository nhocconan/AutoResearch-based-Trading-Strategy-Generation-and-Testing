#!/usr/bin/env python3
"""
Experiment #1060: 1h Primary + 4h/12h HTF — Relaxed Regime + RSI Mean Reversion

Hypothesis: After 768+ failed experiments, the #1 failure mode is 0 trades (Sharpe=0.000).
1h strategies keep failing because entry conditions are TOO STRICT. This strategy:

1. RELAXED ENTRY THRESHOLDS (key lesson from failures):
   - RSI: 30/70 extremes (NOT 20/80 or 25/75)
   - CHOP: 50/60 regime boundaries (NOT 55/65) - wider transition zone
   - NO session filter (too restrictive for 1h)
   - NO volume filter (too restrictive)

2. MULTI-TIMEFRAME STRUCTURE:
   - 4h HMA21: Primary trend direction (aligned with shift(1))
   - 12h HMA21: Macro confirmation filter
   - 1h indicators: Entry timing only

3. REGIME-SWITCHING:
   - CHOP > 60 = RANGE → mean reversion (RSI extremes)
   - CHOP < 50 = TREND → follow HTF HMA direction
   - 50-60 = TRANSITION → hold existing positions

4. HOLD LOGIC (critical for trade count):
   - Once in position, hold until clear exit signal
   - Don't flip-flop on every bar
   - This reduces churn and ensures trades complete

5. POSITION SIZING:
   - BASE_SIZE = 0.25 (conservative for 1h)
   - Discrete levels: 0.0, ±0.25
   - ATR trailing stop: 2.5x from entry

Target: 40-80 trades/year, Sharpe > 0.612, ALL symbols positive
Timeframe: 1h (lower TF needs stricter filters to avoid fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_relaxed_regime_rsi_4h12h_hma_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index - measures market ranging vs trending.
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending.
    We use relaxed thresholds: >60 = range, <50 = trend.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        sum_atr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 1e-10:
            chop[i] = 100 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_rsi(close, period=14):
    """Relative Strength Index with proper min_periods."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    
    avg_gain = gain_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = loss_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi[period:] = 100 - (100 / (1 + rs[period:]))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility and stops."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(series, period):
    """Hull Moving Average - faster response than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_bb(close, period=20, std_mult=2.0):
    """Bollinger Bands for mean reversion reference."""
    n = len(close)
    middle = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    if n < period:
        return upper, middle, lower
    
    close_series = pd.Series(close)
    rolling_mean = close_series.rolling(window=period, min_periods=period).mean()
    rolling_std = close_series.rolling(window=period, min_periods=period).std()
    
    middle = rolling_mean.values
    upper = (rolling_mean + std_mult * rolling_std).values
    lower = (rolling_mean - std_mult * rolling_std).values
    
    return upper, middle, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # === LOAD HTF DATA ONCE BEFORE LOOP (Rule 1 - CRITICAL) ===
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMAs
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # === CALCULATE PRIMARY (1h) INDICATORS ===
    chop = calculate_choppiness_index(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_middle, bb_lower = calculate_bb(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    bars_in_trade = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(chop[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        
        # === REGIME DETECTION (Relaxed thresholds) ===
        is_range = chop[i] > 60.0
        is_trend = chop[i] < 50.0
        is_transition = not is_range and not is_trend
        
        # === HTF TREND BIAS ===
        hma4h_bull = close[i] > hma_4h_aligned[i]
        hma4h_bear = close[i] < hma_4h_aligned[i]
        hma12h_bull = close[i] > hma_12h_aligned[i]
        hma12h_bear = close[i] < hma_12h_aligned[i]
        
        # Strong bias when both HTFs agree
        strong_bull = hma4h_bull and hma12h_bull
        strong_bear = hma4h_bear and hma12h_bear
        
        desired_signal = 0.0
        
        # === RANGE MODE: MEAN REVERSION (relaxed RSI) ===
        if is_range:
            # Long: RSI < 35 + price near BB lower + HTF not strongly bearish
            if rsi[i] < 35 and close[i] <= bb_lower[i] * 1.005 and not strong_bear:
                desired_signal = BASE_SIZE
            # Short: RSI > 65 + price near BB upper + HTF not strongly bullish
            elif rsi[i] > 65 and close[i] >= bb_upper[i] * 0.995 and not strong_bull:
                desired_signal = -BASE_SIZE
            # Weaker signals with HTF confirmation
            elif rsi[i] < 30 and strong_bull:
                desired_signal = BASE_SIZE
            elif rsi[i] > 70 and strong_bear:
                desired_signal = -BASE_SIZE
        
        # === TREND MODE: FOLLOW HTF DIRECTION ===
        elif is_trend:
            # Long: HTF bullish + RSI not overbought
            if strong_bull and rsi[i] < 70:
                desired_signal = BASE_SIZE
            elif hma4h_bull and rsi[i] < 65:
                desired_signal = BASE_SIZE
            # Short: HTF bearish + RSI not oversold
            elif strong_bear and rsi[i] > 30:
                desired_signal = -BASE_SIZE
            elif hma4h_bear and rsi[i] > 35:
                desired_signal = -BASE_SIZE
        
        # === TRANSITION ZONE: HOLD EXISTING ===
        elif is_transition:
            # Don't initiate new trades, but hold existing
            if in_position:
                desired_signal = position_side * BASE_SIZE
        
        # === STOPLOSS (Trailing ATR 2.5x) ===
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
        
        # === HOLD LOGIC (critical for trade count) ===
        # If already in position, hold unless clear exit signal
        if in_position and not stoploss_triggered:
            if position_side > 0:
                # Hold long unless: strong bearish reversal OR RSI very overbought
                if not (strong_bear and rsi[i] > 60):
                    if desired_signal == 0.0:
                        desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short unless: strong bullish reversal OR RSI very oversold
                if not (strong_bull and rsi[i] < 40):
                    if desired_signal == 0.0:
                        desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS (relaxed) ===
        if in_position and position_side > 0:
            # Exit if HTF turns strongly bearish AND RSI overbought
            if strong_bear and rsi[i] > 65:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit if HTF turns strongly bullish AND RSI oversold
            if strong_bull and rsi[i] < 35:
                desired_signal = 0.0
        
        # === DISCRETIZE ===
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
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
                bars_in_trade = 0
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
                bars_in_trade = 0
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
            bars_in_trade += 1
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
                bars_in_trade = 0
        
        signals[i] = desired_signal
    
    return signals