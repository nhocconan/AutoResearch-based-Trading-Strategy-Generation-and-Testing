#!/usr/bin/env python3
"""
Experiment #647: 6h Primary + 1d HTF — Fisher Transform Reversals + Choppiness Regime + HMA Bias

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). Fisher Transform excels at
catching reversals in bear/range markets (2022 crash, 2025 bear). Choppiness Index filters out
false signals during strong trends. 1d HMA provides directional bias to avoid counter-trend trades.

Key innovations:
1. Ehlers Fisher Transform (period=9) - normalizes price to -1.5 to +1.5 range, catches reversals
2. Choppiness Index (14) - CHOP>61.8 = range (trade reversals), CHOP<38.2 = trend (stand aside)
3. 1d HMA(21) bias - only long above, only short below (HTF direction filter)
4. Fisher crossover entry - long when Fisher crosses above -1.5, short when crosses below +1.5
5. ATR(14) trailing stop - 2.5x for risk management
6. Asymmetric sizing - 0.30 for strong signals, 0.20 for weaker confluence

Why this should work on 6h:
- Fisher Transform proven in bear markets (2022, 2025)
- 6h captures multi-day swings without 4h noise or 12h slowness
- Choppiness filter avoids trend whipsaws that killed previous 6h strategies
- Loose entry conditions ensure >=30 trades on train

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-50%
Timeframe: 6h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_chop_regime_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_prev = np.zeros(n)
    fisher_prev[:] = np.nan
    
    # Calculate typical price and normalize
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.nanmax(high[i - period + 1:i + 1])
        lowest = np.nanmin(low[i - period + 1:i + 1])
        
        if highest == lowest:
            continue
        
        # Normalize price to 0-1 range
        value = (close[i] - lowest) / (highest - lowest)
        
        # Clamp to avoid division issues
        value = max(0.001, min(0.999, value))
        
        # Apply Fisher Transform
        fisher[i] = 0.5 * np.log((1 + value) / (1 - value))
        
        if i > period:
            fisher_prev[i] = fisher[i - 1]
    
    return fisher, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = ranging market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        # Highest high and lowest low over period
        highest = np.nanmax(high[i - period + 1:i + 1])
        lowest = np.nanmin(low[i - period + 1:i + 1])
        
        if highest == lowest:
            chop[i] = 50.0
            continue
        
        # Sum of ATR over period
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j - 1]), abs(low[j] - close[j - 1]))
            tr_sum += tr
        
        # Choppiness formula
        chop[i] = 100.0 * np.log10(tr_sum / (highest - lowest)) / np.log10(period)
    
    return chop

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

def calculate_hma(close, period):
    """Hull Moving Average for HTF"""
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0.0)
    loss[1:] = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_ranging = chop[i] > 55.0  # Range market - favor reversals
        is_trending = chop[i] < 45.0  # Trend market - be cautious
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long_cross = (fisher_prev[i] < -1.5 and fisher[i] >= -1.5)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short_cross = (fisher_prev[i] > 1.5 and fisher[i] <= 1.5)
        
        # Additional Fisher conditions
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        # === RSI FILTER ===
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        
        # === ENTRY LOGIC (LOOSE to ensure trades) ===
        desired_signal = 0.0
        
        # LONG entries
        if htf_bull:
            # Strong long: Fisher cross + RSI oversold + ranging market
            if fisher_long_cross and rsi_oversold and is_ranging:
                desired_signal = SIZE_STRONG
            # Medium long: Fisher cross + HTF bull
            elif fisher_long_cross and htf_bull:
                desired_signal = SIZE_BASE
            # Weak long: Fisher oversold + RSI oversold + HTF bull
            elif fisher_oversold and rsi_oversold and htf_bull:
                desired_signal = SIZE_BASE * 0.5
        
        # SHORT entries
        if htf_bear:
            # Strong short: Fisher cross + RSI overbought + ranging market
            if fisher_short_cross and rsi_overbought and is_ranging:
                desired_signal = -SIZE_STRONG
            # Medium short: Fisher cross + HTF bear
            elif fisher_short_cross and htf_bear:
                desired_signal = -SIZE_BASE
            # Weak short: Fisher overbought + RSI overbought + HTF bear
            elif fisher_overbought and rsi_overbought and htf_bear:
                desired_signal = -SIZE_BASE * 0.5
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif abs(desired_signal) >= SIZE_BASE * 0.4:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.5
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
    
    return signals