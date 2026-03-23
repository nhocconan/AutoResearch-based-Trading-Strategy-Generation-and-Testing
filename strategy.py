#!/usr/bin/env python3
"""
Experiment #782: 12h Primary + 1d/1w HTF — HMA Trend + Donchian Breakout + RSI Filter

Hypothesis: After 530+ failed strategies, the pattern is clear:
1. CRSI + Bollinger mean reversion fails on 12h (too many false signals in trends)
2. Complex ADX hysteresis regime detection creates conflicting conditions → 0 trades
3. 12h needs simpler logic: trend-following with breakout entries works better
4. HMA responds faster than EMA for trend detection (less lag on 12h)
5. Donchian(20) breakout + RSI filter reduces false breakouts
6. 1w HTF for major trend bias (avoid counter-trend trades in strong trends)
7. Fewer entry conditions = more trades (avoid the 0-trade failure mode)

Strategy design:
1. 12h HMA(21) for primary trend (faster than EMA)
2. 1d HMA(50) aligned for intermediate trend confirmation
3. 1w HMA(50) aligned for major trend bias (avoid counter-trend)
4. 12h Donchian(20) breakout for entries
5. 12h RSI(14) filter (40-60 neutral, <35 oversold long, >65 overbought short)
6. 12h ATR(14) for trailing stop (3.0x for 12h timeframe)
7. Discrete signals: 0.0, ±0.25, ±0.30

Key differences from failed strategies:
- Removed CRSI (failed in #771, #772, #775, #776, #778, #781)
- Removed ADX hysteresis (adds complexity, failed in #764)
- Removed Bollinger mean reversion (fails in trending markets)
- Simpler: HMA trend + Donchian breakout + RSI filter
- 1w HTF for major trend (new addition)
- Target: 25-40 trades/year on 12h timeframe

Timeframe: 12h (target 20-50 trades/year, lower fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_donchian_rsi_1d1w_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(series, period):
    """
    Hull Moving Average - faster response than EMA.
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(series)
    if n < period:
        return np.full(n, np.nan)
    
    s = pd.Series(series)
    
    # WMA helper
    def wma(data, window):
        weights = np.arange(1, window + 1)
        return data.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(s, half_period)
    wma_full = wma(s, period)
    
    # HMA calculation
    hma_raw = 2 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_period)
    
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

def calculate_donchian(high, low, period=20):
    """Donchian Channels - highest high and lowest low over period."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    if n < period:
        return upper, lower
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

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

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, 21)
    rsi_12h = calculate_rsi(close, 14)
    atr_12h = calculate_atr(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Calculate and align HTF HMAs for trend confirmation
    hma_1d_raw = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, 50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
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
        if np.isnan(hma_12h[i]) or np.isnan(rsi_12h[i]) or np.isnan(atr_12h[i]):
            continue
        if atr_12h[i] <= 1e-10:
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        # === TREND BIAS (Multi-TF HMA) ===
        # 12h HMA21 for immediate trend
        trend_12h_bullish = close[i] > hma_12h[i]
        trend_12h_bearish = close[i] < hma_12h[i]
        
        # 1d HMA50 for intermediate trend
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # 1w HMA50 for major trend (avoid counter-trend)
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === RSI FILTER ===
        rsi_neutral = 40 <= rsi_12h[i] <= 60
        rsi_oversold = rsi_12h[i] < 35
        rsi_overbought = rsi_12h[i] > 65
        rsi_extreme_oversold = rsi_12h[i] < 25
        rsi_extreme_overbought = rsi_12h[i] > 75
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i] * 0.998  # near or above upper
        breakout_short = close[i] < donchian_lower[i] * 1.002  # near or below lower
        
        # === TREND STRENGTH (HMA slope approximation) ===
        hma_slope_long = False
        hma_slope_short = False
        
        if i >= 5 and not np.isnan(hma_12h[i-5]):
            hma_slope_long = hma_12h[i] > hma_12h[i-5]
            hma_slope_short = hma_12h[i] < hma_12h[i-5]
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Primary: 12h bullish + Donchian breakout + RSI not overbought
        if trend_12h_bullish and breakout_long and not rsi_overbought:
            # Strong signal: all TFs aligned bullish
            if trend_1d_bullish and trend_1w_bullish:
                desired_signal = BASE_SIZE
            # Moderate signal: 12h + 1d aligned
            elif trend_1d_bullish:
                desired_signal = REDUCED_SIZE
            # Weak signal: only 12h bullish (but 1w neutral/bullish)
            elif not trend_1w_bearish:
                desired_signal = REDUCED_SIZE
        
        # Pullback long: 12h bullish + RSI oversold + price near Donchian lower
        if trend_12h_bullish and rsi_oversold:
            if close[i] < donchian_lower[i] * 1.05:  # near lower band
                if not trend_1w_bearish:
                    desired_signal = max(desired_signal, REDUCED_SIZE)
        
        # === SHORT ENTRY CONDITIONS ===
        # Primary: 12h bearish + Donchian breakdown + RSI not oversold
        if trend_12h_bearish and breakout_short and not rsi_oversold:
            # Strong signal: all TFs aligned bearish
            if trend_1d_bearish and trend_1w_bearish:
                desired_signal = -BASE_SIZE
            # Moderate signal: 12h + 1d aligned
            elif trend_1d_bearish:
                desired_signal = -REDUCED_SIZE
            # Weak signal: only 12h bearish (but 1w neutral/bearish)
            elif not trend_1w_bullish:
                desired_signal = -REDUCED_SIZE
        
        # Pullback short: 12h bearish + RSI overbought + price near Donchian upper
        if trend_12h_bearish and rsi_overbought:
            if close[i] > donchian_upper[i] * 0.95:  # near upper band
                if not trend_1w_bullish:
                    desired_signal = min(desired_signal, -REDUCED_SIZE)
        
        # === STOPLOSS CHECK (Trailing ATR 3.0x for 12h) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 12h trend still bullish and RSI not extreme overbought
                if trend_12h_bullish and rsi_12h[i] < 80:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 12h trend still bearish and RSI not extreme oversold
                if trend_12h_bearish and rsi_12h[i] > 20:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 12h trend reverses bearish
            if trend_12h_bearish and rsi_12h[i] > 60:
                desired_signal = 0.0
            # Exit if RSI extreme overbought
            if rsi_extreme_overbought:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 12h trend reverses bullish
            if trend_12h_bullish and rsi_12h[i] < 40:
                desired_signal = 0.0
            # Exit if RSI extreme oversold
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
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
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