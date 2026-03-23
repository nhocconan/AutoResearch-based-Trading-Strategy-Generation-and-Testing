#!/usr/bin/env python3
"""
Experiment #959: 4h Primary + 1d HTF — Donchian Breakout + HMA Trend + RSI Timing

Hypothesis: After 664 failed strategies, the key is SIMPLICITY + guaranteed trades.
Complex regime switching causes 0 trades (see #948, #952, #958). 

This strategy uses:
1. 1d HMA(21) for macro trend bias (long only above, short only below)
2. 4h Donchian(20) breakout for clear entry signals
3. 4h RSI(14) for entry timing (avoid chasing - wait for pullback)
4. ATR(14) trailing stop for risk management
5. Simple logic that GUARANTEES trades on all symbols

Why this should work:
- Donchian breakout is proven (Turtle Trading, 50+ year track record)
- 1d HMA filter prevents counter-trend trades in strong trends
- RSI pullback entry avoids buying tops/selling bottoms
- 4h timeframe = 20-50 trades/year target (low fee drag)
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize churn

Critical improvements vs failed strategies:
- RELAXED RSI thresholds (35/65 not 25/75) to ensure entries
- Donchian breakout is PRIMARY signal (not secondary filter)
- No complex regime switching (chop index caused 0 trades)
- Funding rate is OPTIONAL confluence (not required)
- Hold logic maintains position through minor pullbacks

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_rsi_1d_trend_atr_v1"
timeframe = "4h"
leverage = 1.0

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
    """Donchian Channel - upper/lower bounds."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    mid = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
        mid[i] = (upper[i] + lower[i]) / 2
    
    return upper, lower, mid

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

def calculate_keltner(high, low, close, atr_period=14, atr_mult=1.5):
    """Keltner Channel - volatility-based bands."""
    n = len(close)
    middle = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    if n < atr_period + 1:
        return middle, upper, lower
    
    # EMA for middle line
    middle = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    atr = calculate_atr(high, low, close, atr_period)
    
    for i in range(len(close)):
        if not np.isnan(middle[i]) and not np.isnan(atr[i]):
            upper[i] = middle[i] + atr_mult * atr[i]
            lower[i] = middle[i] - atr_mult * atr[i]
    
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
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    keltner_mid, keltner_upper, keltner_lower = calculate_keltner(high, low, close, atr_period=14, atr_mult=1.5)
    
    # Calculate and align 1d HMA for macro trend bias
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
    
    # Breakout tracking to avoid duplicate entries
    last_long_breakout = -100
    last_short_breakout = -100
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(keltner_mid[i]):
            continue
        
        # === MACRO TREND BIAS (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Long breakout: price crosses above Donchian upper
        long_breakout = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        # Short breakout: price crosses below Donchian lower
        short_breakout = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === RSI TIMING (avoid chasing breakouts) ===
        rsi_oversold = rsi_4h[i] < 35
        rsi_overbought = rsi_4h[i] > 65
        rsi_neutral = 35 <= rsi_4h[i] <= 65
        rsi_bullish = rsi_4h[i] > 50
        rsi_bearish = rsi_4h[i] < 50
        
        # === KELTNER POSITION ===
        keltner_long = close[i] > keltner_mid[i]
        keltner_short = close[i] < keltner_mid[i]
        
        # === VOLATILITY CHECK (avoid low vol false breakouts) ===
        vol_expansion = True  # Default allow all
        if i > 30:
            atr_ratio = atr_4h[i] / np.nanmean(atr_4h[i-30:i]) if not np.isnan(atr_4h[i-30:i]).all() else 1.0
            vol_expansion = atr_ratio > 0.8  # Allow if vol not collapsing
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        if macro_bull and vol_expansion:
            # Primary: Donchian breakout + RSI not overbought + trend confirmation
            if long_breakout and rsi_4h[i] < 70 and (i - last_long_breakout > 20):
                desired_signal = BASE_SIZE
                last_long_breakout = i
            # Secondary: RSI pullback in uptrend (buy dip)
            elif rsi_oversold and keltner_long and (i - last_long_breakout > 15):
                desired_signal = REDUCED_SIZE
            # Tertiary: Keltner bounce in bull trend
            elif close[i] > keltner_lower[i] and close[i-1] <= keltner_lower[i-1] and rsi_bullish:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY CONDITIONS ===
        if macro_bear and vol_expansion:
            # Primary: Donchian breakdown + RSI not oversold + trend confirmation
            if short_breakout and rsi_4h[i] > 30 and (i - last_short_breakout > 20):
                desired_signal = -BASE_SIZE
                last_short_breakout = i
            # Secondary: RSI rally in downtrend (sell rip)
            elif rsi_overbought and keltner_short and (i - last_short_breakout > 15):
                desired_signal = -REDUCED_SIZE
            # Tertiary: Keltner rejection in bear trend
            elif close[i] < keltner_upper[i] and close[i-1] >= keltner_upper[i-1] and rsi_bearish:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL MACRO (price ~ HMA) - reduce size or stay flat ===
        if not macro_bull and not macro_bear:
            # Only take reduced size trades in neutral macro
            if long_breakout and rsi_4h[i] < 65:
                desired_signal = REDUCED_SIZE
            elif short_breakout and rsi_4h[i] > 35:
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
                # Hold long if macro bull and RSI not extreme overbought
                if macro_bull and rsi_4h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro bear and RSI not extreme oversold
                if macro_bear and rsi_4h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses to bear + RSI overbought
            if macro_bear and rsi_4h[i] > 70:
                desired_signal = 0.0
            # Exit if price breaks below Keltner mid
            elif close[i] < keltner_mid[i] and rsi_overbought:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses to bull + RSI oversold
            if macro_bull and rsi_4h[i] < 30:
                desired_signal = 0.0
            # Exit if price breaks above Keltner mid
            elif close[i] > keltner_mid[i] and rsi_oversold:
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