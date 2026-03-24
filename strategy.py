#!/usr/bin/env python3
"""
Experiment #579: 1h Primary + 4h/12h HTF — HMA Trend + RSI Pullback + Session Filter

Hypothesis: 1h timeframe with 4h/12h HTF trend confirmation provides optimal balance
between trade frequency and signal quality. Use 4h HMA for trend direction, 1h RSI for
pullback entries, session filter (08-20 UTC) for timing, and Choppiness for regime.

Key lessons from failed experiments (#569-578):
- Too many filters = 0 trades (Sharpe=0.000)
- Entry conditions must be LOOSE enough to generate 40-80 trades/year
- Use OR logic for some conditions, not AND for everything
- Session filter naturally limits trade frequency

Strategy logic:
1. 12h HMA(21) = macro trend bias (slow filter)
2. 4h HMA(21) = medium trend direction (primary filter)
3. 1h RSI(14) = entry timing on pullbacks (RSI<45 long, RSI>55 short)
4. 1h Choppiness(14) = regime (wider thresholds: >50 range, <50 trend)
5. Session filter: 08-20 UTC only for entries (limits trades naturally)
6. ATR(14)*2.5 stoploss on all positions

Entry conditions (LOOSE to ensure trades):
- LONG: 4h HMA bullish + 12h HMA neutral/bull + RSI(14)<45 + session 08-20 UTC
- SHORT: 4h HMA bearish + 12h HMA neutral/bear + RSI(14)>55 + session 08-20 UTC
- Add Choppiness as secondary confirmation (not required)

Target: Sharpe>0.40, trades>=120 train (30/year), trades>=15 test
Timeframe: 1h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h12h_session_v1"
timeframe = "1h"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppy vs trending
    CHOP > 50 = range-bound (mean reversion favored)
    CHOP < 50 = trending (trend follow favored)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for medium trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for macro trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 1h indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC only for entries) ===
        # open_time is in milliseconds, convert to hour
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === HTF BIAS (12h macro + 4h medium) ===
        htf_bull = close[i] > hma_4h_aligned[i] and close[i] > hma_12h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i] and close[i] < hma_12h_aligned[i]
        htf_neutral_4h = close[i] > hma_4h_aligned[i]  # 4h only bullish
        htf_neutral_4h_bear = close[i] < hma_4h_aligned[i]  # 4h only bearish
        
        # === RSI PULLBACK ===
        rsi_pullback_long = rsi[i] < 45.0  # Pullback in uptrend
        rsi_pullback_short = rsi[i] > 55.0  # Pullback in downtrend
        rsi_extreme_long = rsi[i] < 35.0  # Strong oversold
        rsi_extreme_short = rsi[i] > 65.0  # Strong overbought
        
        # === CHOPPINESS REGIME ===
        chop_trend = chop[i] < 50.0  # Trending
        chop_range = chop[i] >= 50.0  # Range
        
        # === ENTRY LOGIC (LOOSE to ensure trades) ===
        desired_signal = 0.0
        
        # LONG entries (multiple conditions with OR logic)
        if in_session:
            # Primary: 4h bullish + RSI pullback
            if htf_neutral_4h and rsi_pullback_long:
                desired_signal = SIZE_BASE
            # Strong: 4h + 12h bullish + RSI pullback
            elif htf_bull and rsi_pullback_long:
                desired_signal = SIZE_STRONG
            # Extreme RSI override (even if HTF neutral)
            elif rsi_extreme_long and htf_neutral_4h:
                desired_signal = SIZE_BASE * 0.8
        
        # SHORT entries (multiple conditions with OR logic)
        if in_session:
            # Primary: 4h bearish + RSI pullback
            if htf_neutral_4h_bear and rsi_pullback_short:
                desired_signal = -SIZE_BASE
            # Strong: 4h + 12h bearish + RSI pullback
            elif htf_bear and rsi_pullback_short:
                desired_signal = -SIZE_STRONG
            # Extreme RSI override (even if HTF neutral)
            elif rsi_extreme_short and htf_neutral_4h_bear:
                desired_signal = -SIZE_BASE * 0.8
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
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
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
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