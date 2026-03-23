#!/usr/bin/env python3
"""
Experiment #253: 1d Primary + 1w HTF — Dual Regime with Choppiness Filter

Hypothesis: Daily timeframe with weekly trend bias + Choppiness Index regime detection
will produce higher-quality trades with lower frequency. Key insights from research:
- CHOP < 38.2 = trending regime → follow HMA trend direction
- CHOP > 61.8 = ranging regime → mean revert at RSI extremes
- 1w HMA(21) for macro bias (proven in best strategies)
- 1d HMA(16/48) crossover for entry timing
- RSI(14) for pullback entries in trend, extremes in range
- ATR(14) 3.0x trailing stoploss

TARGET: 20-50 trades/year on 1d, Sharpe > 0.5 on ALL symbols
POSITION SIZE: 0.25-0.30 discrete (minimize fee churn on daily bars)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_chop_hma_rsi_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

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
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest - lowest
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0, posinf=50.0, neginf=50.0)
    chop = np.clip(chop, 0.0, 100.0)
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Calculate 1w HMA for macro trend (aligned properly with shift(1))
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
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
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 1d TREND (HMA crossover) ===
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # === CHOPPINESS REGIME ===
        is_trending = chop_14[i] < 38.2
        is_ranging = chop_14[i] > 61.8
        is_neutral = (chop_14[i] >= 38.2) and (chop_14[i] <= 61.8)
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # TRENDING REGIME: Follow trend direction
        if is_trending:
            # Long: 1w bullish + 1d bullish + RSI pullback (40-55)
            if price_above_hma_1w and hma_bullish and (40.0 <= rsi_14[i] <= 55.0):
                desired_signal = POSITION_SIZE_FULL
            # Short: 1w bearish + 1d bearish + RSI pullback (45-60)
            elif price_below_hma_1w and hma_bearish and (45.0 <= rsi_14[i] <= 60.0):
                desired_signal = -POSITION_SIZE_FULL
        
        # RANGING REGIME: Mean revert at extremes
        elif is_ranging:
            # Long: RSI oversold (<30) + price near support
            if rsi_14[i] < 30.0:
                desired_signal = POSITION_SIZE_FULL
            # Short: RSI overbought (>70) + price near resistance
            elif rsi_14[i] > 70.0:
                desired_signal = -POSITION_SIZE_FULL
        
        # NEUTRAL REGIME: No trades (wait for clear signal)
        elif is_neutral:
            desired_signal = 0.0
        
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
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 1d trend turns bearish AND 1w turns bearish
        if in_position and position_side > 0 and hma_bearish and price_below_hma_1w:
            desired_signal = 0.0
        
        # Exit short if 1d trend turns bullish AND 1w turns bullish
        if in_position and position_side < 0 and hma_bullish and price_above_hma_1w:
            desired_signal = 0.0
        
        # === RSI EXTREME EXIT (take profit in trending regime) ===
        if is_trending and in_position:
            # Exit long if RSI becomes overbought (>75)
            if position_side > 0 and rsi_14[i] > 75.0:
                desired_signal = 0.0
            # Exit short if RSI becomes oversold (<25)
            elif position_side < 0 and rsi_14[i] < 25.0:
                desired_signal = 0.0
        
        # === RSI MEAN EXIT (take profit in ranging regime) ===
        if is_ranging and in_position:
            # Exit long at RSI midpoint
            if position_side > 0 and rsi_14[i] > 55.0:
                desired_signal = 0.0
            # Exit short at RSI midpoint
            elif position_side < 0 and rsi_14[i] < 45.0:
                desired_signal = 0.0
        
        # === HOLD LOGIC - maintain position if setup still valid ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend still valid
                if (is_trending and hma_bullish and price_above_hma_1w) or \
                   (is_ranging and rsi_14[i] <= 55.0):
                    desired_signal = POSITION_SIZE_HALF
            elif position_side < 0:
                # Hold short if trend still valid
                if (is_trending and hma_bearish and price_below_hma_1w) or \
                   (is_ranging and rsi_14[i] >= 45.0):
                    desired_signal = -POSITION_SIZE_HALF
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
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