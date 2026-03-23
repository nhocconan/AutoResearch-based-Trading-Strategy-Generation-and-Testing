#!/usr/bin/env python3
"""
Experiment #201: 4h Primary + 1d HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: Simplify from #199 by replacing complex CRSI with standard RSI(14) and adding
Donchian Channel breakouts for clearer trend entry signals. The CRSI computation was
complex and may have been overfitting. Donchian breakouts are proven trend-following
signals that work well in crypto markets.

Key components:
1. Donchian(20) breakout - clean trend entry signal (price breaks 20-bar high/low)
2. HMA(21) - fast trend direction with minimal lag
3. RSI(14) filter - avoid entering at extremes (RSI 35-65 zone for trend entries)
4. Choppiness(14) - regime detection (range: mean revert, trend: breakout)
5. 1d HMA - macro directional bias (only trade with HTF trend)

Changes from #199:
1. RSI(14) instead of CRSI - simpler, more robust
2. Donchian breakout entries - clearer trend signals
3. HMA instead of KAMA - faster response to trend changes
4. Looser RSI thresholds (30-70 instead of CRSI 15-85) for more trades
5. Simpler hold logic - hold while Donchian channel not breached

TARGET: 25-45 trades/year on 4h, Sharpe > 0.5 on ALL symbols
Position sizing: 0.0, ±0.25, ±0.30 (discrete to minimize fee churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_rsi_chop_regime_1d_v1"
timeframe = "4h"
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

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Much faster response than EMA with minimal lag.
    """
    n = len(close)
    hma = np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA helper
    def wma(series, span):
        result = np.zeros(len(series))
        for i in range(span - 1, len(series)):
            weights = np.arange(1, span + 1)
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    close_series = pd.Series(close)
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    # 2*WMA(n/2) - WMA(n)
    diff = 2 * wma_half - wma_full
    
    # WMA of diff with sqrt(n) period
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            prev_close = close[j-1] if j > 0 else close[j]
            tr = max(high[j] - low[j], abs(high[j] - prev_close), abs(low[j] - prev_close))
            tr_sum += tr
        
        range_val = highest - lowest
        if range_val < 1e-10 or tr_sum < 1e-10:
            chop[i] = 50.0
        else:
            chop[i] = 100.0 * np.log10(tr_sum / range_val) / np.log10(period)
    
    return chop

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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bounds)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    hma_21 = calculate_hma(close, period=21)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate 1d HMA for macro trend (aligned properly)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === HTF MACRO BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness) ===
        is_range = chop_14[i] > 55.0  # Ranging market
        is_trend = chop_14[i] < 45.0  # Trending market
        # Neutral zone 45-55: hold current bias
        
        # === DONCHIAN BREAKOUT DETECTION ===
        breakout_long = close[i] > donchian_upper[i-1]  # Break above previous upper
        breakout_short = close[i] < donchian_lower[i-1]  # Break below previous lower
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        if is_range:
            # MEAN REVERSION MODE (RSI extremes at range bounds)
            # Long: RSI < 35 + price near Donchian lower + with 1d trend or neutral
            if rsi_14[i] < 35 and close[i] < (donchian_lower[i] * 1.02):
                if price_above_hma_1d:
                    new_signal = POSITION_SIZE_FULL  # With trend
                else:
                    new_signal = POSITION_SIZE_HALF  # Counter-trend, smaller size
            
            # Short: RSI > 65 + price near Donchian upper + with 1d trend or neutral
            elif rsi_14[i] > 65 and close[i] > (donchian_upper[i] * 0.98):
                if price_below_hma_1d:
                    new_signal = -POSITION_SIZE_FULL  # With trend
                else:
                    new_signal = -POSITION_SIZE_HALF  # Counter-trend, smaller size
        
        elif is_trend:
            # TREND FOLLOWING MODE (Donchian breakout + RSI filter)
            # Long: Donchian breakout + RSI not overbought (< 70) + with 1d trend
            if breakout_long and rsi_14[i] < 70:
                if price_above_hma_1d:
                    new_signal = POSITION_SIZE_FULL
                else:
                    new_signal = POSITION_SIZE_HALF
            
            # Short: Donchian breakout + RSI not oversold (> 30) + with 1d trend
            elif breakout_short and rsi_14[i] > 30:
                if price_below_hma_1d:
                    new_signal = -POSITION_SIZE_FULL
                else:
                    new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and conditions still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if RSI not overbought and no Donchian lower break
                if rsi_14[i] < 75 and close[i] > donchian_lower[i]:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if RSI not oversold and no Donchian upper break
                if rsi_14[i] > 25 and close[i] < donchian_upper[i]:
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
        # Exit long if price crosses below 1d HMA (macro trend changed)
        if in_position and position_side > 0 and price_below_hma_1d:
            new_signal = 0.0
        
        # Exit short if price crosses above 1d HMA (macro trend changed)
        if in_position and position_side < 0 and price_above_hma_1d:
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