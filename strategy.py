#!/usr/bin/env python3
"""
Experiment #969: 4h Primary + 1d HTF — Simplified HMA Trend + RSI Pullback + Donchian Confirmation

Hypothesis: After 697 failed strategies, the key insight is SIMPLICITY. Complex regime switching
and too many confluence filters lead to 0 trades (Sharpe=0.000). The proven pattern is:
HTF trend filter + LTF pullback entry. This worked in #964 (Sharpe=0.177) but needs optimization.

Key changes from failed experiments:
1. SIMPLER entry logic — fewer conditions = more trades (avoid Sharpe=0.000)
2. 1d HMA21 for macro trend (not 12h+1d complexity)
3. 4h HMA21 for medium trend + RSI(14) for pullback timing
4. Donchian(20) breakout confirmation for trend strength
5. ATR(14) trailing stop at 2.5x for risk management
6. LOOSE entry thresholds to guarantee 30+ trades/year

Why this should work:
- 4h timeframe targets 25-40 trades/year (optimal fee/trade balance)
- 1d HMA filter prevents counter-trend trades in strong moves
- RSI pullback entries catch dips in uptrends / rallies in downtrends
- Donchian confirmation ensures we're not entering in weak trends
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_donchian_1d_trend_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index — standard Wilder's formula."""
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
    """Hull Moving Average — faster response than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Average True Range — Wilder's smoothing."""
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
    """Donchian Channels — highest high / lowest low over period."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, lower, middle

def calculate_keltner(high, low, close, atr_period=14, atr_mult=2.0, ema_period=20):
    """Keltner Channels — EMA +/- ATR multiple."""
    n = len(close)
    middle = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    if n < max(ema_period, atr_period) + 1:
        return middle, upper, lower
    
    ema = pd.Series(close).ewm(span=ema_period, min_periods=ema_period, adjust=False).mean().values
    atr = calculate_atr(high, low, close, atr_period)
    
    for i in range(n):
        if not np.isnan(ema[i]) and not np.isnan(atr[i]):
            middle[i] = ema[i]
            upper[i] = ema[i] + atr_mult * atr[i]
            lower[i] = ema[i] - atr_mult * atr[i]
    
    return middle, upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    rsi_4h = calculate_rsi(close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    
    hma_4h_raw = calculate_hma(close, 21)
    
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    
    keltner_mid, keltner_upper, keltner_lower = calculate_keltner(high, low, close, atr_period=14, atr_mult=2.0, ema_period=20)
    
    # Calculate and align 1d HMA for macro trend (Rule 2 - use align_htf_to_ltf)
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_raw[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(keltner_mid[i]):
            continue
        
        # === MACRO TREND (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM TREND (4h HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_raw[i]
        trend_4h_bearish = close[i] < hma_4h_raw[i]
        
        # === HMA SLOPE (trend strength) ===
        hma_slope_bull = False
        hma_slope_bear = False
        if i >= 5 and not np.isnan(hma_4h_raw[i-5]):
            hma_slope_bull = hma_4h_raw[i] > hma_4h_raw[i-5]
            hma_slope_bear = hma_4h_raw[i] < hma_4h_raw[i-5]
        
        # === DONCHIAN BREAKOUT CONFIRMATION ===
        donchian_breakout_up = close[i] > donchian_mid[i]
        donchian_breakout_down = close[i] < donchian_mid[i]
        
        # === KELTNER POSITION ===
        keltner_position = (close[i] - keltner_lower[i]) / (keltner_upper[i] - keltner_lower[i]) if (keltner_upper[i] - keltner_lower[i]) > 1e-10 else 0.5
        keltner_lower_touch = close[i] < keltner_lower[i]
        keltner_upper_touch = close[i] > keltner_upper[i]
        
        # === RSI SIGNALS (pullback entries) ===
        rsi_neutral_low = 35 < rsi_4h[i] < 55
        rsi_neutral_high = 45 < rsi_4h[i] < 65
        rsi_oversold = rsi_4h[i] < 45
        rsi_overbought = rsi_4h[i] > 55
        rsi_extreme_oversold = rsi_4h[i] < 35
        rsi_extreme_overbought = rsi_4h[i] > 65
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Primary: Macro bull + 4h bull + RSI pullback (most common, ensures trades)
        if macro_bull and trend_4h_bullish and rsi_oversold:
            desired_signal = BASE_SIZE
        # Secondary: Macro bull + 4h bull + Donchian confirmation + RSI neutral
        elif macro_bull and trend_4h_bullish and donchian_breakout_up and rsi_neutral_low:
            desired_signal = BASE_SIZE
        # Tertiary: Macro bull + Keltner lower touch (dip buy)
        elif macro_bull and keltner_lower_touch and rsi_oversold:
            desired_signal = REDUCED_SIZE
        # Quaternary: Strong HMA slope + RSI not overbought (momentum continuation)
        elif macro_bull and hma_slope_bull and rsi_4h[i] < 60:
            desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY CONDITIONS ===
        # Primary: Macro bear + 4h bear + RSI pullback (most common, ensures trades)
        if macro_bear and trend_4h_bearish and rsi_overbought:
            desired_signal = -BASE_SIZE
        # Secondary: Macro bear + 4h bear + Donchian confirmation + RSI neutral
        elif macro_bear and trend_4h_bearish and donchian_breakout_down and rsi_neutral_high:
            desired_signal = -BASE_SIZE
        # Tertiary: Macro bear + Keltner upper touch (rally sell)
        elif macro_bear and keltner_upper_touch and rsi_overbought:
            desired_signal = -REDUCED_SIZE
        # Quaternary: Strong HMA slope + RSI not oversold (momentum continuation)
        elif macro_bear and hma_slope_bear and rsi_4h[i] > 40:
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
                # Hold long if macro + 4h trend still bull and RSI not extreme overbought
                if macro_bull and trend_4h_bullish and rsi_4h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro + 4h trend still bear and RSI not extreme oversold
                if macro_bear and trend_4h_bearish and rsi_4h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro trend reverses
            if macro_bear and trend_4h_bearish:
                desired_signal = 0.0
            # Exit if RSI extreme overbought (take profit)
            if rsi_extreme_overbought:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro trend reverses
            if macro_bull and trend_4h_bullish:
                desired_signal = 0.0
            # Exit if RSI extreme oversold (take profit)
            if rsi_extreme_oversold:
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
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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