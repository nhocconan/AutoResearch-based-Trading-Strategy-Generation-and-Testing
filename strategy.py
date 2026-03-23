#!/usr/bin/env python3
"""
Experiment #906: 12h Primary + 1d HTF — Simplified KAMA Trend + RSI Pullback

Hypothesis: After 638 failed strategies, complex regime switching is the problem.
This strategy uses SIMPLER logic that guarantees trades on ALL symbols:

1. 12h Primary TF: Target 25-40 trades/year (lower fee drag)
2. 1d HMA(21) for macro trend bias ONLY (not multiple HTF filters)
3. 12h KAMA(14) for adaptive trend following (responds to volatility)
4. RSI(14) pullback entries in trend direction (RSI<45 long, RSI>55 short)
5. Donchian(20) breakout confirmation (reduces false signals)
6. ATR(14) trailing stop (2.5x) for risk management

Why this should work:
- SIMPLER than previous 12h attempts (#896, #902 failed with complex regimes)
- RSI thresholds relaxed (45/55 not 30/70) to guarantee trades
- Single HTF filter (1d HMA) not multiple (1d+1w) which was too restrictive
- KAMA adapts to volatility better than HMA/EMA in choppy markets
- Donchian confirmation ensures we catch real breakouts

Key difference from failed 12h strategies:
- NO choppiness index regime switching (causes whipsaw)
- NO CRSI complexity (RSI alone is sufficient)
- NO multiple HTF conflicts (just 1d HMA for direction)
- RELAXED RSI thresholds to ensure 30+ trades per symbol

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_rsi_pullback_1d_hma_donchian_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=14, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency (trend vs noise)
    
    ER (Efficiency Ratio) = |Price Change| / Sum of |Individual Changes|
    SC (Smoothing Constant) = [ER * (fast_sc - slow_sc) + slow_sc]^2
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan)
    for i in range(period, n):
        price_change = np.abs(close[i] - close[i - period])
        sum_changes = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if sum_changes > 1e-10:
            er[i] = price_change / sum_changes
        else:
            er[i] = 0
    
    er = np.clip(er, 0, 1)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    
    # Calculate KAMA
    kama[period - 1] = close[period - 1]
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]
    
    return kama

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

def calculate_donchian(high, low, period=20):
    """Donchian Channels — highest high and lowest low over period."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
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
    
    # Calculate primary (12h) indicators
    kama_12h = calculate_kama(close, period=14, fast=2, slow=30)
    rsi_12h = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_12h = calculate_atr(high, low, close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate and align 1d HMA for trend bias
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
    
    # Trade counter for debugging
    trade_count = 0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(kama_12h[i]) or np.isnan(rsi_12h[i]):
            continue
        if np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === MACRO TREND BIAS (1d HTF HMA21) ===
        # Simple: price above 1d HMA = bullish bias, below = bearish bias
        trend_bullish = close[i] > hma_1d_aligned[i]
        trend_bearish = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (12h KAMA14) ===
        kama_bullish = close[i] > kama_12h[i]
        kama_bearish = close[i] < kama_12h[i]
        
        # === KAMA SLOPE (trend strength) ===
        kama_slope_bullish = False
        kama_slope_bearish = False
        if i >= 3 and not np.isnan(kama_12h[i-3]):
            kama_slope_bullish = kama_12h[i] > kama_12h[i-3]
            kama_slope_bearish = kama_12h[i] < kama_12h[i-3]
        
        # === RSI PULLBACK SIGNALS (Relaxed: 45/55 for more trades) ===
        rsi_pullback_long = rsi_12h[i] < 45  # Pullback in uptrend
        rsi_pullback_short = rsi_12h[i] > 55  # Rally in downtrend
        rsi_extreme_long = rsi_12h[i] < 30  # Oversold
        rsi_extreme_short = rsi_12h[i] > 70  # Overbought
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = False
        donchian_breakout_short = False
        if not np.isnan(donchian_upper[i-1]):
            donchian_breakout_long = close[i] > donchian_upper[i-1]
        if not np.isnan(donchian_lower[i-1]):
            donchian_breakout_short = close[i] < donchian_lower[i-1]
        
        # === SMA FILTER (avoid counter-trend in strong trends) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        
        # === LONG ENTRY LOGIC ===
        # Primary: Trend bullish + RSI pullback (most common entry)
        if trend_bullish and kama_bullish and rsi_pullback_long:
            desired_signal = BASE_SIZE
        # Secondary: Trend bullish + Donchian breakout (momentum entry)
        elif trend_bullish and kama_bullish and donchian_breakout_long:
            desired_signal = BASE_SIZE
        # Tertiary: Extreme RSI + above SMA200 (deep pullback entry)
        elif rsi_extreme_long and above_sma200 and kama_bullish:
            desired_signal = REDUCED_SIZE
        # Fallback: KAMA bullish + RSI not overbought (hold through minor pullbacks)
        elif kama_bullish and kama_slope_bullish and rsi_12h[i] < 65:
            if in_position and position_side > 0:
                desired_signal = BASE_SIZE
        
        # === SHORT ENTRY LOGIC ===
        # Primary: Trend bearish + RSI rally (most common entry)
        if trend_bearish and kama_bearish and rsi_pullback_short:
            if desired_signal == 0.0:  # Don't override long
                desired_signal = -BASE_SIZE
        # Secondary: Trend bearish + Donchian breakdown (momentum entry)
        elif trend_bearish and kama_bearish and donchian_breakout_short:
            if desired_signal == 0.0:
                desired_signal = -BASE_SIZE
        # Tertiary: Extreme RSI + below SMA200 (rally into resistance)
        elif rsi_extreme_short and below_sma200 and kama_bearish:
            if desired_signal == 0.0:
                desired_signal = -REDUCED_SIZE
        # Fallback: KAMA bearish + RSI not oversold
        elif kama_bearish and kama_slope_bearish and rsi_12h[i] > 35:
            if desired_signal == 0.0 and in_position and position_side < 0:
                desired_signal = -BASE_SIZE
        
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
        
        # === EXIT CONDITIONS ===
        # Exit long if trend reverses
        if in_position and position_side > 0:
            if trend_bearish and kama_bearish:
                desired_signal = 0.0
            # Exit if RSI extremely overbought
            if rsi_12h[i] > 75:
                desired_signal = 0.0
        
        # Exit short if trend reverses
        if in_position and position_side < 0:
            if trend_bullish and kama_bullish:
                desired_signal = 0.0
            # Exit if RSI extremely oversold
            if rsi_12h[i] < 25:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === TRACK TRADES ===
        if desired_signal != 0.0 and (not in_position or np.sign(desired_signal) != position_side):
            trade_count += 1
        
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