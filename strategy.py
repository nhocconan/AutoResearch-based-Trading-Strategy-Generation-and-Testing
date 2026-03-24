#!/usr/bin/env python3
"""
Experiment #1507: 1d Primary + 1w HTF — Dual Regime HMA + Donchian + RSI

Hypothesis: After analyzing 1100+ failed strategies, the pattern is clear:
1. 1d timeframe is proven to work (best strategy #1497 uses 1d, Sharpe=0.424)
2. Complex filters = 0 trades, but NO filters = whipsaw losses
3. DUAL REGIME works: trend-follow in trending markets, mean-revert in chop
4. 1w HTF provides macro bias without over-filtering
5. Donchian breakouts capture momentum, HMA confirms direction, RSI times entry
6. Position size 0.30 appropriate for 1d (20-50 trades/year target)

Key design choices:
- Use 1w HMA(21) for macro trend bias (HTF filter)
- Use 1d HMA(21) for primary trend confirmation
- Use Donchian(20) breakout for momentum entry trigger
- Use RSI(14) for pullback timing within trend (bands: 40-60)
- Use Choppiness Index(14) for regime detection (>50 = chop, <50 = trend)
- Use ATR(14) 2.5x trailing stop for risk management
- Position size 0.30 (discrete: 0.0, ±0.30)
- LOOSE entry conditions to ensure 20-50 trades/year

Timeframe: 1d (as required by experiment)
HTF: 1w (weekly trend bias)
Position Size: 0.30
Target: 80-200 trades/train (4 years), 20-50 trades/test (15 months), Sharpe > 0.618
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_hma_donchian_rsi_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            if np.any(np.isnan(data[i - w_period + 1:i + 1])):
                continue
            result[i] = np.sum(data[i - w_period + 1:i + 1] * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - upper and lower bands"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    > 61.8 = choppy/range, < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    choppiness = np.full(n, np.nan)
    for i in range(period, n):
        if np.isnan(atr[i]):
            continue
        atr_sum = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            choppiness[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return choppiness

def calculate_sma(close, period=50):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    hma_1d = calculate_hma(close, period=21)
    sma_50 = calculate_sma(close, period=50)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    choppiness = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # Appropriate size for 1d (20-50 trades/year target)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_1d[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(choppiness[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1w HMA) - primary direction bias ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND (1d HMA) - confirmation ===
        daily_bull = close[i] > hma_1d[i]
        daily_bear = close[i] < hma_1d[i]
        
        # === SMA50 FILTER ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = choppiness[i] > 50.0  # Range market
        is_trending = choppiness[i] < 50.0  # Trending market
        
        # === RSI CONDITIONS - LOOSE bands for trades ===
        rsi_neutral_long = 40.0 <= rsi[i] <= 60.0
        rsi_neutral_short = 40.0 <= rsi[i] <= 60.0
        rsi_weak_long = rsi[i] < 50.0
        rsi_weak_short = rsi[i] > 50.0
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] >= donchian_upper[i] * 0.998  # Near upper band
        donchian_breakout_short = close[i] <= donchian_lower[i] * 1.002  # Near lower band
        
        # === DESIRED SIGNAL - DUAL REGIME ===
        desired_signal = 0.0
        
        # LONG entries
        if is_trending:
            # Trend-following mode: breakout + trend alignment
            if weekly_bull and daily_bull and donchian_breakout_long:
                desired_signal = BASE_SIZE
            elif weekly_bull and daily_bull and above_sma50 and rsi_weak_long:
                desired_signal = BASE_SIZE * 0.8
            elif weekly_bull and daily_bull and rsi_neutral_long:
                desired_signal = BASE_SIZE * 0.6
        else:
            # Mean-reversion mode: buy near Donchian lower in uptrend
            if weekly_bull and daily_bull and close[i] <= donchian_lower[i] * 1.01:
                desired_signal = BASE_SIZE * 0.7
            elif weekly_bull and rsi[i] < 45.0 and above_sma50:
                desired_signal = BASE_SIZE * 0.5
        
        # SHORT entries
        if is_trending:
            # Trend-following mode: breakdown + trend alignment
            if weekly_bear and daily_bear and donchian_breakout_short:
                desired_signal = -BASE_SIZE
            elif weekly_bear and daily_bear and below_sma50 and rsi_weak_short:
                desired_signal = -BASE_SIZE * 0.8
            elif weekly_bear and daily_bear and rsi_neutral_short:
                desired_signal = -BASE_SIZE * 0.6
        else:
            # Mean-reversion mode: sell near Donchian upper in downtrend
            if weekly_bear and daily_bear and close[i] >= donchian_upper[i] * 0.99:
                desired_signal = -BASE_SIZE * 0.7
            elif weekly_bear and rsi[i] > 55.0 and below_sma50:
                desired_signal = -BASE_SIZE * 0.5
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.7:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE * 0.8
        elif desired_signal >= BASE_SIZE * 0.3:
            final_signal = BASE_SIZE * 0.6
        elif desired_signal <= -BASE_SIZE * 0.7:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.5:
            final_signal = -BASE_SIZE * 0.8
        elif desired_signal <= -BASE_SIZE * 0.3:
            final_signal = -BASE_SIZE * 0.6
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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
        
        signals[i] = final_signal
    
    return signals