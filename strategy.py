#!/usr/bin/env python3
"""
Experiment #205: 1h Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Choppiness Regime

Hypothesis: Simplify from failed #195/#200 (0 trades) while keeping proven elements from #199.
The 1h timeframe failed before because filters were TOO STRICT (session + volume + extreme RSI).
This experiment uses:

1. 4h HMA(21) for macro trend direction (proven in successful 4h strategies)
2. 1h RSI(14) for entry timing with LOOSER thresholds (35/65 not 20/80)
3. Choppiness Index(14) for regime detection (>50 range, <40 trend)
4. 1d HMA for ultimate macro bias filter
5. ATR(14) trailing stoploss at 2.5x

Key changes from failed #195/#200:
1. REMOVED session filter (was killing trade frequency)
2. REMOVED volume filter (was filtering valid signals)
3. LOOSER RSI thresholds (35/65 instead of 25/75)
4. Simpler hold logic - hold while regime valid
5. Position sizing: 0.25 base, 0.30 with HTF confluence

TARGET: 40-70 trades/year on 1h, Sharpe > 0.4 on ALL symbols
Position sizing: 0.0, ±0.25, ±0.30 (discrete to minimize fee churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_chop_regime_4h1d_v1"
timeframe = "1h"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma_half - wma_full
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

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
    """Calculate Choppiness Index (CHOP)."""
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_sum += tr
        
        range_val = highest - lowest
        if range_val < 1e-10 or tr_sum < 1e-10:
            chop[i] = 50.0
        else:
            chop[i] = 100.0 * np.log10(tr_sum / range_val) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Calculate 4h HMA for intermediate trend (aligned properly)
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 1d HMA for macro trend (aligned properly)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_FULL = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]):
            continue
        if np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # === HTF MACRO BIAS ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness) ===
        is_range = chop_14[i] > 50.0  # Ranging market
        is_trend = chop_14[i] < 40.0  # Trending market
        # Neutral zone 40-50: use trend logic
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        if is_range:
            # MEAN REVERSION MODE (RSI extremes with HTF bias)
            # Long: RSI < 35 + price above 4h HMA (with trend) or neutral
            if rsi_14[i] < 35:
                if price_above_hma_4h:
                    new_signal = POSITION_SIZE_FULL  # With intermediate trend
                elif price_above_hma_1d:
                    new_signal = POSITION_SIZE_BASE  # With macro trend only
                else:
                    new_signal = POSITION_SIZE_BASE * 0.5  # Counter-trend, small
            
            # Short: RSI > 65 + price below 4h HMA (with trend) or neutral
            elif rsi_14[i] > 65:
                if price_below_hma_4h:
                    new_signal = -POSITION_SIZE_FULL  # With intermediate trend
                elif price_below_hma_1d:
                    new_signal = -POSITION_SIZE_BASE  # With macro trend only
                else:
                    new_signal = -POSITION_SIZE_BASE * 0.5  # Counter-trend, small
        
        elif is_trend:
            # TREND FOLLOWING MODE (HMA position + RSI pullback)
            # Long: Price above 4h HMA + RSI pullback (40-55)
            if price_above_hma_4h and 40.0 <= rsi_14[i] <= 55.0:
                if price_above_hma_1d:
                    new_signal = POSITION_SIZE_FULL  # Both HTF confirm
                else:
                    new_signal = POSITION_SIZE_BASE  # Only 4h confirms
            
            # Short: Price below 4h HMA + RSI pullback (45-60)
            elif price_below_hma_4h and 45.0 <= rsi_14[i] <= 60.0:
                if price_below_hma_1d:
                    new_signal = -POSITION_SIZE_FULL  # Both HTF confirm
                else:
                    new_signal = -POSITION_SIZE_BASE  # Only 4h confirms
        
        else:
            # NEUTRAL ZONE (40-50 chop): use simpler trend logic
            if price_above_hma_4h and rsi_14[i] < 55:
                new_signal = POSITION_SIZE_BASE
            elif price_below_hma_4h and rsi_14[i] > 45:
                new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and conditions still reasonably valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if RSI not overbought and price above 4h HMA
                if rsi_14[i] < 70 and price_above_hma_4h:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if RSI not oversold and price below 4h HMA
                if rsi_14[i] > 30 and price_below_hma_4h:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if highest_since_entry == 0.0:
                highest_since_entry = close[i]
            else:
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
        
        # === MACRO TREND REVERSAL EXIT ===
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
                if position_side > 0:
                    highest_since_entry = close[i]
                    lowest_since_entry = 0.0
                else:
                    highest_since_entry = 0.0
                    lowest_since_entry = close[i]
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                if position_side > 0:
                    highest_since_entry = close[i]
                    lowest_since_entry = 0.0
                else:
                    highest_since_entry = 0.0
                    lowest_since_entry = close[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals