#!/usr/bin/env python3
"""
Experiment #898: 30m Primary + 4h/1d HTF — Simplified Regime + RSI Pullback

Hypothesis: After 600+ failed strategies, 30m timeframe NEEDS relaxed entry conditions
to generate trades while using HTF for direction. Key lessons from failures:

1. 30m strategies #888, #890, #895 ALL failed with Sharpe=0.000 (0 trades)
2. Too many filters = 0 trades (session, volume, multiple confluence)
3. CRSI is too complex for 30m — use simpler RSI(14) with relaxed thresholds
4. HTF (4h/1d) for DIRECTION, 30m for ENTRY TIMING only
5. RELAXED thresholds: RSI<35/>65 not RSI<20/>80

Strategy Design:
- 4h HMA(21): Primary trend direction (long bias if price > 4h HMA)
- 1d HMA(21): Macro regime filter (avoid counter-trend trades)
- 30m RSI(14): Entry timing (oversold in uptrend, overbought in downtrend)
- 30m ATR(14): Stoploss at 2.5x ATR
- Simple regime: price vs 4h HMA determines long/short bias
- RELAXED entries to guarantee 30+ trades per symbol

Why this should work on 30m:
- Fewer filters = more trades (learned from #888, #890, #895 failures)
- HTF direction filter prevents whipsaw trades
- RSI(14) simpler and more reliable than CRSI on lower TF
- Discrete signal sizes (0.0, ±0.20, ±0.30) minimize fee churn
- Stoploss via signal→0 when price moves 2.5*ATR against position

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 30m (target 40-80 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_pullback_4h1d_hma_atr_v1"
timeframe = "30m"
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands."""
    n = len(close)
    middle = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    if n < period:
        return middle, upper, lower
    
    for i in range(period - 1, n):
        window = close[i-period+1:i+1]
        middle[i] = np.mean(window)
        std = np.std(window, ddof=0)
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
    
    return middle, upper, lower

def calculate_sma_slope(sma_values, lookback=5):
    """Calculate slope of SMA (positive = uptrend, negative = downtrend)."""
    n = len(sma_values)
    slope = np.full(n, np.nan)
    
    for i in range(lookback, n):
        if not np.isnan(sma_values[i]) and not np.isnan(sma_values[i-lookback]):
            slope[i] = (sma_values[i] - sma_values[i-lookback]) / lookback
        else:
            slope[i] = 0.0
    
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (30m) indicators
    rsi_30m = calculate_rsi(close, period=14)
    atr_30m = calculate_atr(high, low, close, period=14)
    bb_middle, bb_upper, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    sma_50_30m = calculate_sma(close, 50)
    sma_200_30m = calculate_sma(close, 200)
    
    # Calculate and align 4h HMA for primary trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro regime
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h HMA slope for trend strength
    hma_4h_slope = calculate_sma_slope(hma_4h_aligned, lookback=8)
    
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
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(rsi_30m[i]) or np.isnan(atr_30m[i]) or atr_30m[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_50_30m[i]) or np.isnan(sma_200_30m[i]):
            continue
        if np.isnan(bb_middle[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === MACRO REGIME (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        trend_4h_slope_positive = hma_4h_slope[i] > 0 if not np.isnan(hma_4h_slope[i]) else False
        trend_4h_slope_negative = hma_4h_slope[i] < 0 if not np.isnan(hma_4h_slope[i]) else False
        
        # === SHORT-TERM TREND FILTER (30m SMA50/200) ===
        above_sma50 = close[i] > sma_50_30m[i]
        below_sma50 = close[i] < sma_50_30m[i]
        above_sma200 = close[i] > sma_200_30m[i]
        below_sma200 = close[i] < sma_200_30m[i]
        
        # === RSI SIGNALS (Relaxed thresholds for more trades) ===
        rsi_oversold = rsi_30m[i] < 40
        rsi_overbought = rsi_30m[i] > 60
        rsi_extreme_oversold = rsi_30m[i] < 30
        rsi_extreme_overbought = rsi_30m[i] > 70
        
        # === BOLLINGER BAND SIGNALS ===
        at_bb_lower = close[i] <= bb_lower[i]
        at_bb_upper = close[i] >= bb_upper[i]
        bb_squeeze = (bb_upper[i] - bb_lower[i]) / bb_middle[i] < 0.05 if bb_middle[i] > 0 else False
        
        desired_signal = 0.0
        
        # === LONG ENTRY LOGIC ===
        # Primary: 4h bullish + RSI oversold pullback
        if trend_4h_bullish and rsi_oversold:
            desired_signal = BASE_SIZE
        
        # Secondary: Macro bull + 4h bullish + RSI extreme oversold (guarantees trades)
        elif macro_bull and trend_4h_bullish and rsi_extreme_oversold:
            desired_signal = BASE_SIZE
        
        # Tertiary: At BB lower + above SMA200 + RSI oversold (mean reversion in uptrend)
        elif at_bb_lower and above_sma200 and rsi_oversold:
            desired_signal = REDUCED_SIZE
        
        # Fallback: RSI extreme oversold + above SMA50 (simple, ensures trades)
        elif rsi_extreme_oversold and above_sma50 and desired_signal == 0:
            desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY LOGIC ===
        # Primary: 4h bearish + RSI overbought pullback
        if trend_4h_bearish and rsi_overbought:
            desired_signal = -BASE_SIZE
        
        # Secondary: Macro bear + 4h bearish + RSI extreme overbought
        elif macro_bear and trend_4h_bearish and rsi_extreme_overbought:
            desired_signal = -BASE_SIZE
        
        # Tertiary: At BB upper + below SMA200 + RSI overbought
        elif at_bb_upper and below_sma200 and rsi_overbought:
            desired_signal = -REDUCED_SIZE
        
        # Fallback: RSI extreme overbought + below SMA50
        elif rsi_extreme_overbought and below_sma50 and desired_signal == 0:
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
                if trend_4h_bullish and rsi_30m[i] < 65:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h trend still bearish and RSI not oversold
                if trend_4h_bearish and rsi_30m[i] > 35:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h trend reverses bearish + RSI overbought
            if trend_4h_bearish and rsi_30m[i] > 65:
                desired_signal = 0.0
            # Exit if macro reverses + RSI overbought
            if macro_bear and rsi_30m[i] > 60:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h trend reverses bullish + RSI oversold
            if trend_4h_bullish and rsi_30m[i] < 35:
                desired_signal = 0.0
            # Exit if macro reverses + RSI oversold
            if macro_bull and rsi_30m[i] < 40:
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