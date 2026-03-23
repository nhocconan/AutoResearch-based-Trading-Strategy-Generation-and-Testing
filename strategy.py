#!/usr/bin/env python3
"""
Experiment #210: 1h Primary + 4h/12h HTF — Fisher Transform + Choppiness Regime + HMA Trend

Hypothesis: Lower timeframe (1h) needs SIMPLER entry logic than 4h to generate adequate trades.
Previous 1h attempts failed due to TOO MANY confluence filters (#200, #205, #208 all Sharpe=0 or negative).

Key insight from failures:
- #200 (1h Fisher+Chop+HMA): 0 trades — filters too strict
- #205 (1h HMA+RSI+Chop): Negative Sharpe — wrong regime logic
- #208 (30m CRSI+Session): 0 trades — session filter killed all signals

NEW APPROACH for 1h:
1. Fisher Transform (period=9) for entry timing — crosses -1.0/+1.0 (NOT extreme -1.5/+1.5)
2. Choppiness Index for regime — CHOP<45=trend, CHOP>55=range
3. 4h HMA(21) for directional bias — slope determines long/short preference
4. 12h ADX(14) for trend strength — ADX>15 confirms trend (not 25, too strict)
5. NO session filter — was killing trades in #208
6. NO volume filter — was killing trades in previous experiments

Entry Logic:
- TREND regime (CHOP<45): Follow 4h HMA direction, Fisher confirms
- RANGE regime (CHOP>55): Mean revert against 4h HMA, Fisher extremes
- NEUTRAL (45-55): Hold current position, no new entries

Position Sizing:
- Full size (0.25): With HTF trend
- Half size (0.15): Counter-trend in range
- Stoploss: 2.5x ATR trailing

TARGET: 40-80 trades/year on 1h, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_chop_hma_adx_regime_4h12h_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    adx = np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        plus_dm_sum = np.sum(plus_dm[i-period+1:i+1])
        minus_dm_sum = np.sum(minus_dm[i-period+1:i+1])
        
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_sum / atr[i]
            minus_di[i] = 100.0 * minus_dm_sum / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period:] = pd.Series(dx[period:]).ewm(span=period, min_periods=period, adjust=False).mean().values[period:]
    
    return adx

def calculate_fisher(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X))
    Where X = 0.67 * (price - lowest) / (highest - lowest) - 0.33
    
    Long signal: Fisher crosses above -1.0
    Short signal: Fisher crosses below +1.0
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        range_val = highest - lowest
        if range_val < 1e-10:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
        else:
            x = 0.67 * ((close[i] - lowest) / range_val - 0.33)
            x = np.clip(x, -0.999, 0.999)  # Prevent division by zero in ln
            fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x + 1e-10))
        
        fisher_signal[i] = fisher[i-1] if i > 0 else 0.0
    
    return fisher, fisher_signal

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 1h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    fisher, fisher_signal = calculate_fisher(high, low, close, period=9)
    
    # Calculate 4h HMA for directional bias (aligned properly)
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 4h HMA slope (trend direction)
    hma_4h_slope = np.zeros(n)
    for i in range(2, n):
        if not np.isnan(hma_4h_aligned[i]) and not np.isnan(hma_4h_aligned[i-1]):
            hma_4h_slope[i] = hma_4h_aligned[i] - hma_4h_aligned[i-1]
    
    # Calculate 12h ADX for trend strength (aligned properly)
    adx_12h_raw = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, period=14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.25
    POSITION_SIZE_HALF = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(200, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_4h_slope[i]):
            continue
        if np.isnan(adx_12h_aligned[i]):
            continue
        
        # === HTF MACRO BIAS (4h HMA) ===
        hma_bullish = hma_4h_slope[i] > 0  # 4h HMA sloping up
        hma_bearish = hma_4h_slope[i] < 0  # 4h HMA sloping down
        
        # === TREND STRENGTH (12h ADX) ===
        adx_strong = adx_12h_aligned[i] > 15.0  # Trending (lowered from 25)
        adx_weak = adx_12h_aligned[i] <= 15.0  # Weak trend / range
        
        # === REGIME DETECTION (Choppiness) ===
        is_range = chop_14[i] > 55.0  # Ranging market
        is_trend = chop_14[i] < 45.0  # Trending market
        is_neutral = 45.0 <= chop_14[i] <= 55.0  # Transition zone
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_bull_cross = (fisher_signal[i] < -1.0) and (fisher[i] >= -1.0)  # Cross above -1.0
        fisher_bear_cross = (fisher_signal[i] > 1.0) and (fisher[i] <= 1.0)  # Cross below +1.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        if is_trend and adx_strong:
            # TREND FOLLOWING MODE
            # Long: 4h HMA bullish + Fisher bull cross
            if hma_bullish and fisher_bull_cross:
                new_signal = POSITION_SIZE_FULL
            
            # Short: 4h HMA bearish + Fisher bear cross
            elif hma_bearish and fisher_bear_cross:
                new_signal = -POSITION_SIZE_FULL
        
        elif is_range or adx_weak:
            # MEAN REVERSION MODE
            # Long: Fisher deeply oversold (< -1.2) regardless of HMA
            if fisher[i] < -1.2:
                if hma_bullish:
                    new_signal = POSITION_SIZE_FULL  # With HTF trend
                else:
                    new_signal = POSITION_SIZE_HALF  # Counter-trend, smaller
            
            # Short: Fisher deeply overbought (> +1.2) regardless of HMA
            elif fisher[i] > 1.2:
                if hma_bearish:
                    new_signal = -POSITION_SIZE_FULL  # With HTF trend
                else:
                    new_signal = -POSITION_SIZE_HALF  # Counter-trend, smaller
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and Fisher hasn't reversed
        if in_position and new_signal == 0.0 and not is_neutral:
            if position_side > 0:
                # Hold long if Fisher not overbought
                if fisher[i] < 1.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if Fisher not oversold
                if fisher[i] > -1.0:
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
        
        # === REGIME CHANGE EXIT ===
        # Exit if regime changes against position
        if in_position and is_neutral:
            # Neutral zone: reduce position or exit
            if position_side > 0 and fisher[i] > 0.5:
                new_signal = 0.0
            elif position_side < 0 and fisher[i] < -0.5:
                new_signal = 0.0
        
        # === HTF TREND REVERSAL EXIT ===
        # Exit long if 4h HMA turns bearish
        if in_position and position_side > 0 and hma_bearish and fisher[i] > 0:
            new_signal = 0.0
        
        # Exit short if 4h HMA turns bullish
        if in_position and position_side < 0 and hma_bullish and fisher[i] < 0:
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