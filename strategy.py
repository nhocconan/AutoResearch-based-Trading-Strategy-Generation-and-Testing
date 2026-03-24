#!/usr/bin/env python3
"""
Experiment #547: 6h Primary + 1d/1w HTF — SuperTrend + Fisher Transform + HMA Alignment

Hypothesis: 6h timeframe with SuperTrend provides cleaner trend signals than HMA/EMA during
volatile periods. SuperTrend uses ATR for dynamic stops, reducing whipsaw. Fisher Transform
(period=9) catches reversals at extremes (-1.5/+1.5 levels) with high precision in bear markets.
Combined with 1d/1w HMA alignment for macro bias, this should outperform RSI-based 6h strategies.

Key differences from failed #540 (6h_hma_rsi_pullback):
1. SuperTrend instead of HMA - ATR-based stops, clearer trend signals
2. Fisher Transform instead of RSI - better reversal detection at extremes
3. Dual HTF: 1d HMA for medium bias + 1w HMA for macro bias
4. Entry requires ALL three: SuperTrend flip + Fisher extreme + HTF alignment
5. Simpler exit: SuperTrend flip OR 2.5 ATR stoploss

Strategy logic:
1. 1w HMA(21) = macro trend bias (very slow filter)
2. 1d HMA(21) = medium trend bias
3. 6h SuperTrend(10, 3.0) = trend direction with ATR stops
4. 6h Fisher Transform(9) = reversal timing (crosses -1.5/+1.5)
5. 6h ADX(14) = trend strength filter (ADX>22 = valid trend)
6. ATR(14)*2.5 hard stoploss on all positions

Regime-adaptive entries:
- LONG: SuperTrend flips green + Fisher < -1.5 crossing up + price > 1d HMA + price > 1w HMA
- SHORT: SuperTrend flips red + Fisher > +1.5 crossing down + price < 1d HMA + price < 1w HMA
- FLAT: Any condition not met OR ADX < 22 (weak trend)

Target: Sharpe>0.40, trades>=30 train (7.5/year), trades>=3 test
Timeframe: 6h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_supertrend_fisher_hma_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range using Wilder's smoothing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    SuperTrend indicator
    Returns: supertrend values, direction (1=bull, -1=bear)
    
    Upper Band = (High + Low) / 2 + multiplier * ATR
    Lower Band = (High + Low) / 2 - multiplier * ATR
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    supertrend[:] = np.nan
    direction = np.zeros(n)  # 1 = bull (price above ST), -1 = bear (price below ST)
    
    hl2 = (high + low) / 2.0
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize
    supertrend[period] = upper_band[period]
    direction[period] = -1 if close[period] < supertrend[period] else 1
    
    for i in range(period + 1, n):
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            supertrend[i] = np.nan
            direction[i] = 0
            continue
        
        # Calculate basic bands
        basic_upper = upper_band[i]
        basic_lower = lower_band[i]
        
        # Determine final SuperTrend value
        if direction[i-1] == 1:  # Previous was bullish
            if close[i] > supertrend[i-1]:
                # Stay bullish, use lower band
                supertrend[i] = max(basic_lower, supertrend[i-1])
                direction[i] = 1
            else:
                # Flip to bearish
                supertrend[i] = basic_upper
                direction[i] = -1
        else:  # Previous was bearish
            if close[i] < supertrend[i-1]:
                # Stay bearish, use upper band
                supertrend[i] = min(basic_upper, supertrend[i-1])
                direction[i] = -1
            else:
                # Flip to bullish
                supertrend[i] = basic_lower
                direction[i] = 1
    
    return supertrend, direction

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform
    Transforms price into a Gaussian normal distribution
    Entry signals when Fisher crosses -1.5 (long) or +1.5 (short)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_prev = np.zeros(n)
    fisher_prev[:] = np.nan
    
    # Normalize price to -1 to +1 range
    for i in range(period - 1, n):
        highest = np.nanmax(high[i-period+1:i+1]) if 'high' in dir() else np.nanmax(close[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1]) if 'low' in dir() else np.nanmin(close[i-period+1:i+1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            fisher[i] = 0.0
            fisher_prev[i] = fisher[i-1] if i > 0 else 0.0
            continue
        
        # Normalize close to 0-1, then to -1 to +1
        normalized = (close[i] - lowest) / price_range
        normalized = 0.999 * (2.0 * normalized - 1.0)  # Keep within -0.999 to +0.999
        
        # Apply Fisher transform
        if abs(normalized) < 0.999:
            fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        else:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
        
        fisher_prev[i] = fisher[i-1] if i > 0 else 0.0
    
    return fisher, fisher_prev

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_hma(close, period):
    """Hull Moving Average"""
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

def generate_signals(prices):
    global high, low  # Make available for Fisher calculation
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for medium trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    supertrend, st_direction = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    fisher, fisher_prev = calculate_fisher(close, period=9)
    adx = calculate_adx(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    prev_st_direction = 0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(supertrend[i]) or np.isnan(fisher[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w macro + 1d medium) ===
        htf_bull = close[i] > hma_1d_aligned[i] and close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i] and close[i] < hma_1w_aligned[i]
        
        # === SUPERTREND DIRECTION ===
        st_bull = st_direction[i] == 1
        st_bear = st_direction[i] == -1
        
        # SuperTrend flip detection
        st_flip_bull = st_direction[i] == 1 and prev_st_direction == -1
        st_flip_bear = st_direction[i] == -1 and prev_st_direction == 1
        
        # === FISHER TRANSFORM ===
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_cross_up = fisher_prev[i] < -1.5 and fisher[i] >= -1.5
        fisher_cross_down = fisher_prev[i] > 1.5 and fisher[i] <= 1.5
        
        # === ADX TREND STRENGTH ===
        adx_valid = adx[i] > 22.0  # Minimum trend strength
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: SuperTrend flip bull + Fisher cross up + HTF bull + ADX valid
        if st_flip_bull and fisher_cross_up and htf_bull and adx_valid:
            desired_signal = SIZE_STRONG
        # LONG: Already bullish + Fisher recovery from oversold
        elif st_bull and htf_bull and fisher_oversold and fisher[i] > fisher_prev[i] and adx_valid:
            desired_signal = SIZE_BASE
        
        # SHORT: SuperTrend flip bear + Fisher cross down + HTF bear + ADX valid
        elif st_flip_bear and fisher_cross_down and htf_bear and adx_valid:
            desired_signal = -SIZE_STRONG
        # SHORT: Already bearish + Fisher recovery from overbought
        elif st_bear and htf_bear and fisher_overbought and fisher[i] < fisher_prev[i] and adx_valid:
            desired_signal = -SIZE_BASE
        
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
        
        # === EXIT ON SUPERTREND FLIP AGAINST POSITION ===
        if in_position and position_side > 0 and st_flip_bear:
            desired_signal = 0.0
        if in_position and position_side < 0 and st_flip_bull:
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
        prev_st_direction = st_direction[i]
    
    return signals