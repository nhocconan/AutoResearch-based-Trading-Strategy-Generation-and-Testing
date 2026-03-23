#!/usr/bin/env python3
"""
Experiment #286: 12h Primary + 1d HTF — Regime-Adaptive KAMA/Donchian

Hypothesis: 12h timeframe is slow enough for trend following BUT needs regime detection
to avoid 2022-style whipsaws. Key innovations:
- KAMA (Kaufman Adaptive MA) adapts speed to volatility (slow in chop, fast in trend)
- Choppiness Index (CHOP) switches between trend-follow and mean-revert modes
- 1d HMA(21) for macro bias (only trade with daily trend)
- Donchian(20) breakout in trending regime (CHOP < 38.2)
- RSI(14) extremes in choppy regime (CHOP > 61.8)
- ATR(14) 3x trailing stoploss
- Position size: 0.30 (discrete)

Why this differs from failed #276/#282:
- KAMA instead of HMA (adapts to volatility automatically)
- Regime-adaptive entry logic (not fixed RSI or fixed Donchian)
- 1d HMA as hard filter (not soft bias)
- Wider RSI bands for chop regime (25/75 vs 30/70)

TARGET: 25-40 trades/year on 12h, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_chop_regime_1d_donchian_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency ratio.
    Period: lookback for Efficiency Ratio
    fast_period: fastest smoothing constant (default 2)
    slow_period: slowest smoothing constant (default 30)
    """
    close_s = pd.Series(close)
    n = len(close_s)
    
    # Efficiency Ratio = |change| / sum(|changes|)
    change = np.abs(close_s.diff(period))
    volatility = np.abs(close_s.diff()).rolling(window=period, min_periods=period).sum()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        er = change / (volatility + 1e-10)
    er = er.fillna(0)
    
    # Smoothing constant
    sc = (er * (2.0/(fast_period+1) - 2.0/(slow_period+1)) + 2.0/(slow_period+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[period] = close_s.iloc[period]
    
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close_s.iloc[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
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
    Choppiness Index (CHOP).
    Measures market choppiness vs trending.
    CHOP > 61.8 = choppy/range-bound
    CHOP < 38.2 = trending
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Highest high and lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    # ATR
    atr = calculate_atr(high, low, close, period)
    
    # CHOP formula
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10((hh - ll).values / (atr * np.sqrt(period)) + 1e-10) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bounds)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators (primary timeframe)
    kama_12h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate and align 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.30  # Discrete position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(kama_12h[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d HMA) - HARD FILTER ===
        # Only trade in direction of daily trend
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_trending = chop_14[i] < 38.2  # Trending regime
        is_choppy = chop_14[i] > 61.8   # Choppy/range regime
        is_neutral = not is_trending and not is_choppy
        
        # === KAMA TREND ===
        kama_bullish = close[i] > kama_12h[i]
        kama_bearish = close[i] < kama_12h[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # TRENDING REGIME: Donchian breakout with KAMA confirmation
        if is_trending:
            # Long: Price breaks Donchian upper + above KAMA + daily bullish
            if close[i] > donchian_upper[i] and kama_bullish and price_above_hma_1d:
                desired_signal = POSITION_SIZE
            
            # Short: Price breaks Donchian lower + below KAMA + daily bearish
            elif close[i] < donchian_lower[i] and kama_bearish and price_below_hma_1d:
                desired_signal = -POSITION_SIZE
        
        # CHOPPY REGIME: RSI mean reversion
        elif is_choppy:
            # Long: RSI < 25 (oversold) + daily bullish bias
            if rsi_14[i] < 25.0 and price_above_hma_1d:
                desired_signal = POSITION_SIZE
            
            # Short: RSI > 75 (overbought) + daily bearish bias
            elif rsi_14[i] > 75.0 and price_below_hma_1d:
                desired_signal = -POSITION_SIZE
        
        # NEUTRAL REGIME: No trades (sit out)
        # This reduces whipsaw in transition periods
        
        # === STOPLOSS CHECK (3.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === REGIME CHANGE EXIT ===
        # Exit if regime changes against position
        if in_position and position_side > 0:
            # Long position: exit if regime becomes choppy and RSI > 60
            if is_choppy and rsi_14[i] > 60.0:
                desired_signal = 0.0
            # Exit if daily trend reverses
            elif price_below_hma_1d:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Short position: exit if regime becomes choppy and RSI < 40
            if is_choppy and rsi_14[i] < 40.0:
                desired_signal = 0.0
            # Exit if daily trend reverses
            elif price_above_hma_1d:
                desired_signal = 0.0
        
        # === HOLD LOGIC ===
        # If already in position and no exit signal, maintain position
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and kama_bullish and price_above_hma_1d:
                desired_signal = POSITION_SIZE
            elif position_side < 0 and kama_bearish and price_below_hma_1d:
                desired_signal = -POSITION_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals