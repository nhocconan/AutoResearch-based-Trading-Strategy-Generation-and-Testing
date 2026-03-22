#!/usr/bin/env python3
"""
Experiment #120: 1d KAMA + Supertrend + 1w HMA Trend Filter + ATR Trailing Stop

Hypothesis: Daily timeframe with weekly trend filter provides optimal balance:
- 1d KAMA (Kaufman Adaptive MA) adapts to volatility - fast in trends, slow in chop
- 1d Supertrend (ATR=10, mult=3) provides clear trend direction with stop levels
- 1w HMA(21) gives major trend bias - only trade in direction of weekly trend
- ADX(14) > 20 filters out weak/choppy markets
- ATR(14) trailing stop at 2.5*ATR protects capital during reversals

Why this might beat current best (mtf_4h_kama_1d_hma_adx_atr_v1, Sharpe=0.478):
- 1d timeframe reduces noise and whipsaws vs 4h
- Weekly filter is stronger than daily filter for major trend alignment
- KAMA + Supertrend combination catches trends earlier than either alone
- Fewer but higher-quality trades = less fee drag, better risk-adjusted returns

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_supertrend_1w_hma_adx_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts to market volatility - fast during trends, slow during chop.
    ER (Efficiency Ratio) = |Net Change| / Sum of Absolute Changes
    SC (Smoothing Constant) = [ER * (fast_sc - slow_sc) + slow_sc]^2
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < period + slow:
        return kama
    
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio
    net_change = np.abs(close - np.roll(close, period))
    net_change[:period] = np.nan
    
    sum_changes = pd.Series(np.abs(close - np.roll(close, 1))).rolling(window=period, min_periods=period).sum().values
    sum_changes[:period] = np.nan
    
    er = np.zeros(n)
    er[:] = np.nan
    mask = (sum_changes > 0) & (~np.isnan(net_change))
    er[mask] = net_change[mask] / sum_changes[mask]
    er = np.clip(er, 0, 1)
    
    # Smoothing constants
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        if np.isnan(er[i]):
            kama[i] = kama[i-1]
        else:
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, supertrend_direction (1=long, -1=short)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    supertrend[:] = np.nan
    direction[:] = np.nan
    
    # Calculate basic bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize
    supertrend[period] = upper_band[period]
    direction[period] = -1  # Start short
    
    for i in range(period + 1, n):
        if np.isnan(atr[i]):
            supertrend[i] = np.nan
            direction[i] = np.nan
            continue
        
        # Update bands based on previous direction
        if direction[i-1] == 1:  # Previous was long
            lower_band[i] = max(lower_band[i], supertrend[i-1])
            upper_band[i] = hl2[i] + multiplier * atr[i]
        else:  # Previous was short
            upper_band[i] = min(upper_band[i], supertrend[i-1])
            lower_band[i] = hl2[i] - multiplier * atr[i]
        
        # Determine current direction
        if close[i] > upper_band[i]:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        elif close[i] < lower_band[i]:
            supertrend[i] = upper_band[i]
            direction[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
    
    return supertrend, direction

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    if n < period * 2:
        return adx
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    mask = tr_s > 0
    plus_di[mask] = 100 * plus_dm_s[mask] / tr_s[mask]
    minus_di[mask] = 100 * minus_dm_s[mask] / tr_s[mask]
    
    di_sum = plus_di + minus_di
    mask2 = di_sum > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx_series = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx_series.values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    supertrend, st_direction = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(supertrend[i]) or np.isnan(st_direction[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1w HMA = major trend bias (stronger filter than daily)
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === SUPERTREND DIRECTION ===
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # === KAMA TREND ===
        # Price above KAMA = bullish, below = bearish
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx[i] > 20 if not np.isnan(adx[i]) else False
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Strong: 1w bullish + Supertrend long + KAMA bull + ADX strong
        if bull_trend_1w and st_long and kama_bull and adx_strong:
            new_signal = SIZE_STRONG
        # Moderate: 1w bullish + Supertrend long + KAMA bull
        elif bull_trend_1w and st_long and kama_bull:
            new_signal = SIZE_BASE
        # Weak but valid: 1w bullish + Supertrend long (ensure trades on all symbols)
        elif bull_trend_1w and st_long:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # Strong: 1w bearish + Supertrend short + KAMA bear + ADX strong
        if bear_trend_1w and st_short and kama_bear and adx_strong:
            new_signal = -SIZE_STRONG
        # Moderate: 1w bearish + Supertrend short + KAMA bear
        elif bear_trend_1w and st_short and kama_bear:
            new_signal = -SIZE_BASE
        # Weak but valid: 1w bearish + Supertrend short (ensure trades on all symbols)
        elif bear_trend_1w and st_short:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals