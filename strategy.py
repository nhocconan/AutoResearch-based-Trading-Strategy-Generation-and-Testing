#!/usr/bin/env python3
"""
Experiment #652: 12h Primary + 1d/1w HTF — Multi-Signal Confluence with Regime Adaptation

Hypothesis: 12h timeframe provides optimal balance between signal quality and trade frequency.
Combining Fisher Transform (reversals), KAMA (adaptive trend), Donchian (breakouts), and
Choppiness Index (regime filter) with OR logic for entries ensures sufficient trade generation
across all market conditions (rallies, crashes, ranges).

Key innovations:
1. Multiple independent entry signals (OR logic) — any one can trigger, not all required
2. Looser Fisher thresholds (-1.5/+1.5) to catch more reversals
3. Donchian breakout as backup entry when Fisher doesn't trigger
4. 1d HMA + 1w HMA dual HTF filter for macro bias
5. Choppiness regime switch: mean revert when choppy, trend follow when trending
6. Hold logic maintains positions through minor pullbacks (critical for PnL)
7. Trailing stop at 3.0 ATR to let winners run

Why this should beat Sharpe=0.612:
- 12h TF = fewer false signals than 4h, more trades than 1d
- OR logic entries = guaranteed trade generation on major moves
- Dual HTF (1d + 1w) = better trend filter than single HTF
- Conservative sizing (0.25-0.30) survives 77% crash with ~25% DD
- Hold logic prevents premature exits during trend continuation

Target: Sharpe > 0.612, trades >= 20 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_kama_donchian_chop_dualhtf_v1"
timeframe = "12h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform.
    Converts price to Gaussian normal distribution for clearer reversal signals.
    Long: Fisher crosses above -1.5 from below
    Short: Fisher crosses below +1.5 from above
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period:
        return fisher, fisher_signal
    
    price = np.zeros(n)
    
    for i in range(period, n):
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        range_val = hh - ll
        if range_val < 1e-10:
            range_val = 1e-10
        
        price_raw = (close[i] - ll) / range_val
        
        if i > period:
            price[i] = 0.33 * 2 * (price_raw - 0.5) + 0.67 * price[i-1]
        else:
            price[i] = 0.33 * 2 * (price_raw - 0.5)
        
        price[i] = np.clip(price[i], -0.999, 0.999)
        fisher[i] = 0.5 * np.log((1 + price[i]) / (1 - price[i]))
        fisher_signal[i] = fisher[i-1] if i > 0 else fisher[i]
    
    return fisher, fisher_signal

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency (trend vs noise).
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    er = np.zeros(n)
    
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i+1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    kama[er_period] = np.mean(close[:er_period+1])
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/ranging, CHOP < 38.2 = trending
    We use: > 55 = chop (mean revert), < 45 = trend (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop_raw = 100.0 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(period)
        chop = np.clip(chop_raw, 0, 100)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Hull Moving Average for smoother HTF trend."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_donchian_channels(high, low, period=20):
    """Donchian Channels for breakout detection."""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    return upper, lower, middle

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 12h indicators (primary timeframe)
    fisher_12h, fisher_signal_12h = calculate_fisher_transform(high, low, close, period=9)
    kama_12h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian_channels(high, low, period=20)
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(fisher_12h[i]) or np.isnan(fisher_signal_12h[i]):
            continue
        if np.isnan(kama_12h[i]) or np.isnan(chop_12h[i]):
            continue
        if np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_12h[i] > 55.0
        is_trending = chop_12h[i] < 45.0
        
        # === HTF TREND BIAS (1d + 1w HMA) ===
        htf_1d_bullish = close[i] > hma_1d_aligned[i]
        htf_1d_bearish = close[i] < hma_1d_aligned[i]
        htf_1w_bullish = close[i] > hma_1w_aligned[i]
        htf_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # Strong HTF bias (both 1d and 1w agree)
        htf_strong_bull = htf_1d_bullish and htf_1w_bullish
        htf_strong_bear = htf_1d_bearish and htf_1w_bearish
        
        # === KAMA TREND (12h) ===
        kama_bullish = close[i] > kama_12h[i]
        kama_bearish = close[i] < kama_12h[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_long_cross = (fisher_12h[i] > -1.5) and (fisher_signal_12h[i] <= -1.5)
        fisher_short_cross = (fisher_12h[i] < 1.5) and (fisher_signal_12h[i] >= 1.5)
        
        # Fisher extreme levels
        fisher_oversold = fisher_12h[i] < -1.8
        fisher_overbought = fisher_12h[i] > 1.8
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        donchian_long_breakout = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_short_breakout = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC — OR conditions (any one can trigger) ===
        
        # LONG entries (multiple independent triggers)
        long_signal = False
        
        # Trigger 1: Fisher oversold cross in choppy market
        if is_choppy and fisher_oversold:
            long_signal = True
        
        # Trigger 2: Fisher long cross with KAMA support
        if fisher_long_cross and kama_bullish:
            long_signal = True
        
        # Trigger 3: Donchian breakout with HTF bullish bias
        if donchian_long_breakout and (htf_1d_bullish or htf_1w_bullish):
            long_signal = True
        
        # Trigger 4: KAMA bullish + HTF strong bullish (trend follow)
        if kama_bullish and htf_strong_bull and fisher_12h[i] < 1.0:
            long_signal = True
        
        # Trigger 5: Mean reversion in chop + HTF not bearish
        if is_choppy and fisher_12h[i] < -1.0 and not htf_strong_bear:
            long_signal = True
        
        # SHORT entries (multiple independent triggers)
        short_signal = False
        
        # Trigger 1: Fisher overbought in choppy market
        if is_choppy and fisher_overbought:
            short_signal = True
        
        # Trigger 2: Fisher short cross with KAMA resistance
        if fisher_short_cross and kama_bearish:
            short_signal = True
        
        # Trigger 3: Donchian breakdown with HTF bearish bias
        if donchian_short_breakout and (htf_1d_bearish or htf_1w_bearish):
            short_signal = True
        
        # Trigger 4: KAMA bearish + HTF strong bearish (trend follow)
        if kama_bearish and htf_strong_bear and fisher_12h[i] > -1.0:
            short_signal = True
        
        # Trigger 5: Mean reversion in chop + HTF not bullish
        if is_choppy and fisher_12h[i] > 1.0 and not htf_strong_bull:
            short_signal = True
        
        # Set desired signal based on triggers
        if long_signal and not short_signal:
            desired_signal = SIZE_LONG
        elif short_signal and not long_signal:
            desired_signal = -SIZE_SHORT
        elif long_signal and short_signal:
            # Conflict — use HTF bias to decide
            if htf_strong_bull:
                desired_signal = SIZE_LONG
            elif htf_strong_bear:
                desired_signal = -SIZE_SHORT
            else:
                desired_signal = 0.0  # Stay flat on conflict
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if KAMA still bullish OR Fisher not extremely overbought
                if kama_bullish and fisher_12h[i] < 2.0:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if KAMA still bearish OR Fisher not extremely oversold
                if kama_bearish and fisher_12h[i] > -2.0:
                    desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_LONG
        elif desired_signal < 0:
            desired_signal = -SIZE_SHORT
        
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