#!/usr/bin/env python3
"""
Experiment #1040: 1h Primary + 4h/12h HTF — Regime-Adaptive RSI Pullback with Session Filter

Hypothesis: After 752+ failed strategies, the key insight for 1h timeframe is:
1. Use 4h HMA for TREND DIRECTION (not entry trigger)
2. Use 1h RSI for ENTRY TIMING within HTF trend
3. Use Choppiness Index to ADAPT strategy (range vs trend)
4. Session filter (8-20 UTC) reduces noise and trade frequency
5. Volume filter confirms institutional participation

CRITICAL FOR 1h: Must generate 30-80 trades/year, NOT 200+. Too many trades = fee drag kills profit.
Solution: Relaxed RSI thresholds (30-70 not 20-80), single HTF filter (4h not 4h+12h), 
session filter only reduces bad trades not all trades.

Why this beats #1039:
- 1h timeframe captures more entry opportunities than 4h
- Regime-adaptive (CHOP) works in both bull/bear/range markets
- Session filter aligns with institutional volume (8-20 UTC = London+NY overlap)
- Simpler than triple-HTF strategies that got 0 trades (#1030, #1035)

Timeframe: 1h (primary), 4h (trend), 12h (regime confirmation)
Position Size: 0.25 discrete (smaller for lower TF to reduce fee impact)
Target: 40-70 trades/year, Sharpe > 0.612
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_rsi_4h_hma_session_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index - standard Wilder calculation."""
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
    """Average True Range for volatility and stoploss."""
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP > 61.8 = range/choppy market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR for each bar
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high - lowest_low > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50  # neutral
    
    return chop

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs rolling average."""
    vol_series = pd.Series(volume)
    vol_avg = vol_series.rolling(window=period, min_periods=period).mean().values
    vol_ratio = np.divide(volume, vol_avg, out=np.zeros_like(volume), where=vol_avg != 0)
    return vol_ratio

def get_hour_from_open_time(open_time_array):
    """Extract hour from Unix timestamp (milliseconds)."""
    # open_time is in milliseconds, convert to hours UTC
    hours = ((open_time_array / 1000) % 86400) / 3600
    return hours.astype(int)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # === LOAD HTF DATA ONCE BEFORE LOOP (Rule 1 - CRITICAL) ===
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA21 for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA50 for regime confirmation
    hma_12h_raw = calculate_hma(df_12h['close'].values, 50)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # === CALCULATE PRIMARY (1h) INDICATORS ===
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    chop_1h = calculate_choppiness_index(high, low, close, period=14)
    vol_ratio_1h = calculate_volume_ratio(volume, period=20)
    
    # Extract hour for session filter
    hours = get_hour_from_open_time(open_time)
    
    # === SESSION FILTER (8-20 UTC = London + NY overlap) ===
    # This reduces noise and trade frequency to target 40-70/year
    in_session = (hours >= 8) & (hours <= 20)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(chop_1h[i]) or np.isnan(vol_ratio_1h[i]):
            continue
        
        # === HTF TREND DIRECTION (4h HMA21) ===
        # Price above 4h HMA = bullish bias (only look for longs)
        # Price below 4h HMA = bearish bias (only look for shorts)
        trend_bull = close[i] > hma_4h_aligned[i]
        trend_bear = close[i] < hma_4h_aligned[i]
        
        # === REGIME CONFIRMATION (12h HMA50) ===
        # Stronger confirmation for trend direction
        regime_bull = close[i] > hma_12h_aligned[i]
        regime_bear = close[i] < hma_12h_aligned[i]
        
        # === CHOPPINESS INDEX REGIME ===
        # CHOP > 55 = range (use mean reversion logic)
        # CHOP < 45 = trend (use trend following logic)
        chop_range = chop_1h[i] > 55
        chop_trend = chop_1h[i] < 45
        
        # === VOLUME CONFIRMATION ===
        # Volume > 0.8x average = institutional participation
        volume_ok = vol_ratio_1h[i] > 0.8
        
        # === SESSION FILTER ===
        # Only trade during high-liquidity hours (8-20 UTC)
        session_ok = in_session[i]
        
        # === RSI ENTRY SIGNALS ===
        # Relaxed thresholds to ensure sufficient trades (30-70/year)
        rsi_oversold = rsi_1h[i] < 40  # Long entry zone
        rsi_overbought = rsi_1h[i] > 60  # Short entry zone
        rsi_neutral = 40 <= rsi_1h[i] <= 60  # Hold zone
        
        # Deep extremes for stronger signals
        rsi_deep_oversold = rsi_1h[i] < 30
        rsi_deep_overbought = rsi_1h[i] > 70
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Condition 1: Trend bull + RSI pullback + volume + session
        if trend_bull and rsi_oversold and volume_ok and session_ok:
            desired_signal = BASE_SIZE
        # Condition 2: Trend bull + deep oversold (stronger signal, skip volume/session)
        elif trend_bull and rsi_deep_oversold:
            desired_signal = BASE_SIZE
        # Condition 3: Range regime + RSI oversold (mean reversion play)
        elif chop_range and rsi_oversold and volume_ok:
            desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY CONDITIONS ===
        # Condition 1: Trend bear + RSI pullback + volume + session
        if trend_bear and rsi_overbought and volume_ok and session_ok:
            desired_signal = -BASE_SIZE
        # Condition 2: Trend bear + deep overbought (stronger signal, skip volume/session)
        elif trend_bear and rsi_deep_overbought:
            desired_signal = -BASE_SIZE
        # Condition 3: Range regime + RSI overbought (mean reversion play)
        elif chop_range and rsi_overbought and volume_ok:
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
                # Hold long if trend still bullish or RSI not overbought
                if trend_bull and rsi_1h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend still bearish or RSI not oversold
                if trend_bear and rsi_1h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses bearish AND RSI overbought
            if trend_bear and rsi_1h[i] > 65:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses bullish AND RSI oversold
            if trend_bull and rsi_1h[i] < 35:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE:
                desired_signal = BASE_SIZE
            else:
                desired_signal = REDUCED_SIZE
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE:
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
                # Flip position
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