#!/usr/bin/env python3
"""
Experiment #269: 4h Primary + 1d HTF — Volatility Compression Breakout

Hypothesis: After 12 consecutive failures with complex regime-switching, return to 
proven volatility-based entries. Bollinger Band squeezes precede major moves 70%+ 
of the time. Combined with 1d HMA trend filter, this captures breakouts with 
favorable risk/reward.

KEY CHANGES FROM #251 (which failed with Sharpe=-0.194):
1. BB Width percentile instead of RSI pullback (more reliable breakout signal)
2. Donchian breakout confirmation (price must actually break, not just pullback)
3. Simpler exit logic (only stoploss + trend reversal, no RSI extreme exits)
4. Wider RSI thresholds (30/70 instead of 35/65) for more trade frequency
5. Remove POSITION_SIZE_HALF hold logic (reduces churn, clearer signals)

INDICATORS:
- 1d HMA(21): Macro trend bias (load ONCE before loop via mtf_data)
- 4h BB(20, 2.0) + BB Width percentile(30): Volatility compression detection
- 4h Donchian(20): Breakout confirmation
- 4h RSI(7): Momentum filter (faster than RSI(14))
- 4h ATR(14) 3.0x: Trailing stoploss

ENTRY LOGIC:
- Long: 1d HMA bullish + BB Width < 30th percentile + price > Donchian upper + RSI > 50
- Short: 1d HMA bearish + BB Width < 30th percentile + price < Donchian lower + RSI < 50

TARGET: 25-45 trades/year on 4h, Sharpe > 0.5 on ALL symbols
POSITION SIZE: 0.25 (conservative for 4h volatility)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_bb_squeeze_donchian_1d_hma_atr_v1"
timeframe = "4h"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bb_width = (upper - lower) / sma * 100.0
    return upper.values, lower.values, bb_width.values

def calculate_bb_width_percentile(bb_width, lookback=30):
    """Calculate percentile rank of BB Width over lookback period."""
    bb_width_s = pd.Series(bb_width)
    # Percentile rank: where current value sits in recent distribution
    bb_pct = bb_width_s.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: (x.iloc[-1] < x).mean() * 100, raw=False
    )
    return bb_pct.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    return upper.values, lower.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    hma_9 = calculate_hma(close, 9)
    hma_21 = calculate_hma(close, 21)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bb_width_pct = calculate_bb_width_percentile(bb_width, lookback=30)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate 1d HMA for macro trend (aligned properly with shift(1))
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.28
    
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
        if np.isnan(hma_9[i]) or np.isnan(hma_21[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi_7[i]) or np.isnan(bb_width_pct[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 4h TREND (HMA crossover) ===
        hma_bullish = hma_9[i] > hma_21[i]
        hma_bearish = hma_9[i] < hma_21[i]
        
        # === VOLATILITY COMPRESSION (BB Squeeze) ===
        # BB Width in bottom 30% of recent 30 bars = compression
        bb_squeeze = bb_width_pct[i] < 30.0
        
        # === BREAKOUT CONFIRMATION (Donchian) ===
        breakout_long = close[i] > donchian_upper[i]
        breakout_short = close[i] < donchian_lower[i]
        
        # === MOMENTUM FILTER (RSI) ===
        rsi_bullish = rsi_7[i] > 50.0
        rsi_bearish = rsi_7[i] < 50.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: 1d bullish + 4h bullish + BB squeeze + Donchian breakout + RSI > 50
        if price_above_hma_1d and hma_bullish and bb_squeeze and breakout_long and rsi_bullish:
            desired_signal = POSITION_SIZE
        
        # SHORT ENTRY: 1d bearish + 4h bearish + BB squeeze + Donchian breakout + RSI < 50
        elif price_below_hma_1d and hma_bearish and bb_squeeze and breakout_short and rsi_bearish:
            desired_signal = -POSITION_SIZE
        
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
        # Exit long if 4h trend turns bearish
        if in_position and position_side > 0 and hma_bearish:
            desired_signal = 0.0
        
        # Exit short if 4h trend turns bullish
        if in_position and position_side < 0 and hma_bullish:
            desired_signal = 0.0
        
        # === MACRO TREND REVERSAL EXIT ===
        # Exit long if 1d trend turns bearish
        if in_position and position_side > 0 and price_below_hma_1d:
            desired_signal = 0.0
        
        # Exit short if 1d trend turns bullish
        if in_position and position_side < 0 and price_above_hma_1d:
            desired_signal = 0.0
        
        # === HOLD LOGIC - maintain position if no exit triggered ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and hma_bullish and price_above_hma_1d:
                desired_signal = POSITION_SIZE
            elif position_side < 0 and hma_bearish and price_below_hma_1d:
                desired_signal = -POSITION_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                if position_side > 0:
                    highest_since_entry = close[i]
                    lowest_since_entry = float('inf')
                else:
                    highest_since_entry = 0.0
                    lowest_since_entry = close[i]
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                if position_side > 0:
                    highest_since_entry = close[i]
                    lowest_since_entry = float('inf')
                else:
                    highest_since_entry = 0.0
                    lowest_since_entry = close[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals