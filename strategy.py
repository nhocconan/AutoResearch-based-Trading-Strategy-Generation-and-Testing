#!/usr/bin/env python3
"""
Experiment #1026: 12h Primary + 1d HTF — HMA Trend + Donchian Breakout + RSI Filter

Hypothesis: After analyzing 744+ failed strategies, the pattern is clear:
1. Complex regime-switching (Choppiness + Fisher + CRSI) creates too many conflicting filters → 0 trades
2. Pure mean-reversion fails in strong trends (2021 bull, 2022 crash)
3. Pure trend-following gets whipsawed in ranges (2023-2024)

SOLUTION: Simpler is better for 12h timeframe:
- 1d HMA21 = long-term trend bias (only trade in direction)
- 12h HMA21 = medium-term confirmation
- Donchian(20) breakout = clean entry signal (proven on SOL Sharpe +0.782)
- RSI(14) filter = avoid entering at extremes (RSI 35-65 sweet spot)
- ATR(14) trailing stop = 2.5x for risk management

Why 12h works:
- Target 20-50 trades/year (vs 100+ on lower TF)
- Less fee drag, higher quality signals
- Each trade has more conviction (multiple TF alignment)

Critical fixes from failures:
- RELAXED RSI filter (30-70 not 40-60) to ensure trades generate
- Donchian breakout on EITHER high OR low (not both required)
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn
- Proper HTF alignment via mtf_data helper (NO manual i//N mapping)

Target: Sharpe > 0.612, trades >= 10 per symbol train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_donchian_rsi_1d_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - reduces lag while maintaining smoothness."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_donchian_channels(high, low, period=20):
    """
    Donchian Channels - breakout system.
    Upper = highest high over period
    Lower = lowest low over period
    Breakout above upper = long signal
    Breakout below lower = short signal
    """
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    # Use EMA for RSI calculation (smoother)
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    
    avg_gain = gain_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = loss_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops."""
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

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope direction."""
    n = len(hma_values)
    slope = np.full(n, np.nan)
    
    for i in range(lookback, n):
        if not np.isnan(hma_values[i]) and not np.isnan(hma_values[i-lookback]):
            slope[i] = hma_values[i] - hma_values[i-lookback]
    
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA21 for long-term trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 12h HMA21 for medium-term trend
    hma_12h = calculate_hma(close, 21)
    
    # Calculate 12h Donchian channels (20-period breakout)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, period=20)
    
    # Calculate 12h RSI for entry filter
    rsi_12h = calculate_rsi(close, period=14)
    
    # Calculate 12h ATR for stoploss
    atr_12h = calculate_atr(high, low, close, period=14)
    
    # Calculate HMA slopes for trend confirmation
    hma_12h_slope = calculate_hma_slope(hma_12h, lookback=3)
    hma_1d_slope = calculate_hma_slope(hma_1d_aligned, lookback=3)
    
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
        if np.isnan(hma_12h[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(rsi_12h[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_12h_slope[i]) or np.isnan(hma_1d_slope[i]):
            continue
        
        # === LONG-TERM TREND BIAS (1d HMA21) ===
        # Only long when price > 1d HMA (bullish bias)
        # Only short when price < 1d HMA (bearish bias)
        long_term_bullish = close[i] > hma_1d_aligned[i]
        long_term_bearish = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM-TERM CONFIRMATION (12h HMA21) ===
        medium_bullish = close[i] > hma_12h[i]
        medium_bearish = close[i] < hma_12h[i]
        
        # === HMA SLOPE CONFIRMATION ===
        hma_12h_rising = hma_12h_slope[i] > 0
        hma_12h_falling = hma_12h_slope[i] < 0
        
        # === RSI FILTER (avoid extremes) ===
        # Relaxed range to ensure trades generate: 30-70
        rsi_neutral = 30 <= rsi_12h[i] <= 70
        rsi_oversold = rsi_12h[i] < 40
        rsi_overbought = rsi_12h[i] > 60
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Breakout above upper channel
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        # Breakout below lower channel
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # Also check if price is near channel (within 1% for early entry)
        channel_range = donchian_upper[i] - donchian_lower[i]
        if channel_range > 1e-10:
            near_upper = close[i] > donchian_upper[i] - 0.01 * channel_range
            near_lower = close[i] < donchian_lower[i] + 0.01 * channel_range
        else:
            near_upper = False
            near_lower = False
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Need: long-term bullish + medium bullish + RSI not overbought + breakout
        if long_term_bullish and medium_bullish:
            # Strong breakout with RSI confirmation
            if donchian_breakout_long and rsi_neutral:
                desired_signal = BASE_SIZE
            # Early entry near upper channel with rising HMA
            elif near_upper and hma_12h_rising and rsi_oversold:
                desired_signal = REDUCED_SIZE
            # Pullback entry in uptrend
            elif medium_bullish and rsi_oversold and hma_12h_rising:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY CONDITIONS ===
        # Need: long-term bearish + medium bearish + RSI not oversold + breakout
        if long_term_bearish and medium_bearish:
            # Strong breakout with RSI confirmation
            if donchian_breakout_short and rsi_neutral:
                desired_signal = -BASE_SIZE
            # Early entry near lower channel with falling HMA
            elif near_lower and not hma_12h_rising and rsi_overbought:
                desired_signal = -REDUCED_SIZE
            # Pullback entry in downtrend
            elif medium_bearish and rsi_overbought and hma_12h_falling:
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
        
        # === TREND REVERSAL EXIT ===
        if in_position and position_side > 0:
            # Exit long if medium trend reverses bearish
            if medium_bearish and hma_12h_falling:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if medium trend reverses bullish
            if medium_bullish and hma_12h_rising:
                desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend still bullish
                if long_term_bullish and medium_bullish:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend still bearish
                if long_term_bearish and medium_bearish:
                    desired_signal = -BASE_SIZE
        
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