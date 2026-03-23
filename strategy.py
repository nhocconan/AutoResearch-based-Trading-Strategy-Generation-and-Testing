#!/usr/bin/env python3
"""
Experiment #645: 4h Primary + 1d/1w HTF — Fisher Transform + HMA Trend + Choppiness Regime

Hypothesis: 4h timeframe provides optimal balance between signal quality and trade frequency.
Using 1d HMA for macro trend bias + Fisher Transform for precise entries + Choppiness 
for regime detection. This addresses the 1h failure pattern by using slower timeframe
with stricter HTF confluence.

Key innovations:
1. Fisher Transform (period=9) for reversal entries at extremes (-1.5/+1.5 thresholds)
2. 1d HMA(21) for macro trend bias — only trade in direction of daily trend
3. Choppiness Index regime switch — mean revert when CHOP>55, trend follow when CHOP<45
4. 1w HMA(21) for ultra-macro filter — prevents counter-trend trades in strong moves
5. Volume confirmation — only enter when volume > 0.7x 20-period average
6. ATR trailing stop (2.5x) with hold logic to maintain positions through pullbacks
7. Discrete signal sizing (0.25/0.30) to minimize fee churn

Why this should beat Sharpe=0.612:
- 4h timeframe = proven winner (current best is 4h strategy)
- Fisher Transform catches reversals better than RSI in bear/range markets
- Dual HTF filter (1d + 1w) prevents false breakouts
- Regime-adaptive logic switches between mean-reversion and trend-following
- Conservative sizing (0.25-0.30) survives 77% crash with manageable DD

Target: Sharpe > 0.612, trades >= 30 train, >= 5 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_hma_chop_dualhtf_vol_v1"
timeframe = "4h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform.
    Converts price to a Gaussian normal distribution for clearer reversal signals.
    
    Formula:
    1. Price = (0.33 * 2 * ((close - LL) / (HH - LL) - 0.5) + 0.67 * prev_Price)
    2. Fisher = 0.5 * ln((1 + Price) / (1 - Price))
    
    Long signal: Fisher crosses above -1.5 from below
    Short signal: Fisher crosses below +1.5 from above
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period:
        return fisher, fisher_signal
    
    price = np.zeros(n)
    
    for i in range(period, n):
        # Highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        # Avoid division by zero
        range_val = hh - ll
        if range_val < 1e-10:
            range_val = 1e-10
        
        # Normalized price
        price_raw = (close[i] - ll) / range_val
        
        # Smoothed price (0.33 * current + 0.67 * previous)
        if i > period:
            price[i] = 0.33 * 2 * (price_raw - 0.5) + 0.67 * price[i-1]
        else:
            price[i] = 0.33 * 2 * (price_raw - 0.5)
        
        # Clip to avoid log domain errors
        price[i] = np.clip(price[i], -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + price[i]) / (1 - price[i]))
        
        # Signal line (previous Fisher value)
        fisher_signal[i] = fisher[i-1] if i > 0 else fisher[i]
    
    return fisher, fisher_signal

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
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    # Sum ATR over period
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate CHOP
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard formula."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
        rsi[period:] = 100 - (100 / (1 + rs[period:]))
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 4h indicators (primary timeframe)
    fisher_4h, fisher_signal_4h = calculate_fisher_transform(high, low, close, period=9)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    rsi_4h = calculate_rsi(close, period=14)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    
    for i in range(100, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(fisher_4h[i]) or np.isnan(fisher_signal_4h[i]):
            continue
        if np.isnan(chop_4h[i]) or np.isnan(atr_4h[i]):
            continue
        if np.isnan(rsi_4h[i]) or np.isnan(vol_sma[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if atr_4h[i] <= 1e-10 or vol_sma[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_4h[i] > 55.0
        is_trending = chop_4h[i] < 45.0
        
        # === HTF TREND BIAS (1d HMA) ===
        htf_1d_bullish = close[i] > hma_1d_aligned[i]
        htf_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === ULTRA-HTF TREND BIAS (1w HMA) ===
        htf_1w_bullish = close[i] > hma_1w_aligned[i]
        htf_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.7 * vol_sma[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below
        fisher_long_cross = (fisher_4h[i] > -1.5) and (fisher_signal_4h[i] <= -1.5)
        # Short: Fisher crosses below +1.5 from above
        fisher_short_cross = (fisher_4h[i] < 1.5) and (fisher_signal_4h[i] >= 1.5)
        
        # Fisher extreme levels (for mean reversion in chop)
        fisher_oversold = fisher_4h[i] < -1.8
        fisher_overbought = fisher_4h[i] > 1.8
        
        # RSI extremes
        rsi_oversold = rsi_4h[i] < 35
        rsi_overbought = rsi_4h[i] > 65
        
        desired_signal = 0.0
        
        # === REGIME 1: CHOPPY MARKET (Mean Reversion) ===
        if is_choppy:
            # Long: Fisher oversold + RSI oversold + HTF 1d not strongly bearish + volume
            if fisher_oversold and rsi_oversold and not htf_1d_bearish and volume_confirmed:
                desired_signal = SIZE_LONG
            # Short: Fisher overbought + RSI overbought + HTF 1d not strongly bullish + volume
            elif fisher_overbought and rsi_overbought and not htf_1d_bullish and volume_confirmed:
                desired_signal = -SIZE_SHORT
            # Fisher cross signals in chop with volume
            elif fisher_long_cross and volume_confirmed:
                desired_signal = SIZE_LONG
            elif fisher_short_cross and volume_confirmed:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 2: TRENDING MARKET (Trend Follow) ===
        elif is_trending:
            # Long: HTF 1d bullish + HTF 1w not bearish + Fisher not overbought + volume
            if htf_1d_bullish and not htf_1w_bearish and fisher_4h[i] < 1.0 and volume_confirmed:
                desired_signal = SIZE_LONG
            # Short: HTF 1d bearish + HTF 1w not bullish + Fisher not oversold + volume
            elif htf_1d_bearish and not htf_1w_bullish and fisher_4h[i] > -1.0 and volume_confirmed:
                desired_signal = -SIZE_SHORT
            # Fisher cross with trend confirmation
            elif fisher_long_cross and htf_1d_bullish and volume_confirmed:
                desired_signal = SIZE_LONG
            elif fisher_short_cross and htf_1d_bearish and volume_confirmed:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 3: NEUTRAL/TRANSITION ===
        else:
            # Use Fisher with RSI confirmation and volume
            if fisher_oversold and rsi_oversold and volume_confirmed:
                desired_signal = SIZE_LONG
            elif fisher_overbought and rsi_overbought and volume_confirmed:
                desired_signal = -SIZE_SHORT
            elif fisher_long_cross and volume_confirmed:
                desired_signal = SIZE_LONG
            elif fisher_short_cross and volume_confirmed:
                desired_signal = -SIZE_SHORT
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HTF 1d still bullish OR Fisher not extremely overbought
                if htf_1d_bullish and fisher_4h[i] < 2.0:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if HTF 1d still bearish OR Fisher not extremely oversold
                if htf_1d_bearish and fisher_4h[i] > -2.0:
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
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            # If same side, update trailing stop levels
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