#!/usr/bin/env python3
"""
Experiment #214: 4h Primary + 12h HTF — Dual Regime (Chop Index) + HMA/Donchian + RSI

Hypothesis: Use Choppiness Index to switch between mean-reversion (range) and trend-following
regimes. In range (CHOP>61.8): RSI extremes with HMA filter. In trend (CHOP<38.2): Donchian
breakout with HMA confirmation. 12h HMA provides macro bias filter.

Key innovations vs #204:
1. Stricter RSI thresholds (25/75 vs 35/65) for fewer, higher-quality trades
2. Donchian(20) breakout for trend regime (not tried in #204)
3. 12h HMA macro filter (not 1d) — faster response to regime changes
4. ATR trailing stoploss with signal→0 on breach
5. Position sizing: 0.0, ±0.25, ±0.30 (discrete levels)

TARGET: 25-45 trades/year on 4h, Sharpe > 0.50 on ALL symbols
Position sizing: MAX 0.30 to control drawdown during 2022 crash
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_dual_regime_hma_donchian_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) window
    Faster and smoother than EMA, less lag.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bands)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h indicators (primary timeframe)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate 12h HMA for macro trend (aligned properly)
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            continue
        if np.isnan(rsi_14[i]):
            continue
        if np.isnan(chop_14[i]):
            continue
        if np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range = chop_14[i] > 61.8  # Range/choppy market
        is_trend = chop_14[i] < 38.2  # Trending market
        # Neutral zone: 38.2 <= CHOP <= 61.8 (no new entries)
        
        # === HTF MACRO BIAS (12h HMA) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === TREND DETECTION (4h HMA crossover) ===
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # RANGE REGIME: Mean reversion with RSI extremes
        if is_range:
            # LONG: RSI < 25 (oversold) + price above 12h HMA (macro bias)
            if rsi_14[i] < 25.0 and price_above_hma_12h:
                new_signal = POSITION_SIZE_HALF  # Smaller size in range
            
            # SHORT: RSI > 75 (overbought) + price below 12h HMA (macro bias)
            elif rsi_14[i] > 75.0 and price_below_hma_12h:
                new_signal = -POSITION_SIZE_HALF  # Smaller size in range
        
        # TREND REGIME: Breakout with HMA confirmation
        elif is_trend:
            # LONG: Donchian breakout + HMA bullish + 12h bias
            if close[i] > donchian_upper[i-1] and hma_bullish and price_above_hma_12h:
                new_signal = POSITION_SIZE_FULL
            
            # SHORT: Donchian breakdown + HMA bearish + 12h bias
            elif close[i] < donchian_lower[i-1] and hma_bearish and price_below_hma_12h:
                new_signal = -POSITION_SIZE_FULL
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and regime/trend still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if HMA still bullish or in range with RSI not overbought
                if hma_bullish or (is_range and rsi_14[i] < 70.0):
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if HMA still bearish or in range with RSI not oversold
                if hma_bearish or (is_range and rsi_14[i] > 30.0):
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if HMA crosses bearish (in trend regime)
        if in_position and position_side > 0 and is_trend and hma_bearish:
            new_signal = 0.0
        
        # Exit short if HMA crosses bullish (in trend regime)
        if in_position and position_side < 0 and is_trend and hma_bullish:
            new_signal = 0.0
        
        # Exit if macro trend reverses against position
        if in_position and position_side > 0 and price_below_hma_12h:
            new_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_12h:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals