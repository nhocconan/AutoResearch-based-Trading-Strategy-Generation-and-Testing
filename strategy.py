#!/usr/bin/env python3
"""
Experiment #502: 4h Primary + 1d/1w HTF — Fisher Transform + HMA Trend + RSI

Hypothesis: 4h timeframe with dual HTF bias (1w for long-term, 1d for intermediate)
combined with Fisher Transform reversals and loose RSI entries will generate
20-50 trades/year with Sharpe > 0.40.

Strategy logic:
1. 1w HMA(21) = ultra long-term trend bias (slowest, most stable)
2. 1d HMA(21) = intermediate trend confirmation
3. 4h Fisher Transform(9) = reversal detection (crosses -1.5/+1.5)
4. 4h RSI(14) = entry timing with loose thresholds (35/65)
5. 4h HMA(21) = short-term trend alignment
6. ATR(14)*2.5 stoploss on all positions
7. OR logic for entries (any trigger works, not AND)

Why this might work:
- Fisher Transform excels in bear/range markets (2022 crash, 2025 bear)
- Dual HTF bias prevents whipsaw (both 1w and 1d must agree for strong signals)
- Loose RSI thresholds ensure sufficient trades (35/65 not 30/70)
- 4h timeframe naturally limits trade frequency to 20-50/year

Target: Sharpe>0.40, trades>=80 train (20/year), trades>=12 test (3/year)
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_hma_rsi_1d1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
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
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

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

def calculate_fisher(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Excellent for catching reversals in bear/range markets
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan)
    
    # Calculate typical price
    typical = (high + low + np.roll(high, 1)) / 3.0
    typical[0] = (high[0] + low[0] + high[0]) / 3.0
    
    # Normalize price over lookback period
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            fisher[i] = 0.0
            continue
        
        # Normalize to 0-1 range
        normalized = (typical[i] - lowest) / (highest - lowest)
        
        # Clamp to avoid division by zero
        normalized = np.clip(normalized, 0.001, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Smooth with previous value
        if i > period and not np.isnan(fisher[i-1]):
            fisher[i] = 0.67 * fisher[i] + 0.33 * fisher[i-1]
    
    return fisher

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 4h indicators
    hma_4h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    fisher = calculate_fisher(high, low, period=9)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h[i]) or np.isnan(rsi[i]) or np.isnan(fisher[i]):
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
        
        if np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1w HTF BIAS (ultra long-term) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === 1d HTF BIAS (intermediate) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h HMA TREND ===
        hma_bull = close[i] > hma_4h[i]
        hma_bear = close[i] < hma_4h[i]
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === RSI EXTREMES (LOOSE: 35/65 for entries) ===
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        rsi_extreme_oversold = rsi[i] < 35.0
        rsi_extreme_overbought = rsi[i] > 65.0
        rsi_rising = rsi[i] > rsi[i-1] if i > 0 else False
        rsi_falling = rsi[i] < rsi[i-1] if i > 0 else False
        
        # === FISHER TRANSFORM REVERSALS ===
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_cross_up = fisher[i] > -1.5 and fisher[i-1] <= -1.5 if i > 0 and not np.isnan(fisher[i-1]) else False
        fisher_cross_down = fisher[i] < 1.5 and fisher[i-1] >= 1.5 if i > 0 and not np.isnan(fisher[i-1]) else False
        
        # === HTF CONFLUENCE ===
        # Strong bull: both 1w and 1d agree
        htf_strong_bull = htf_1w_bull and htf_1d_bull
        htf_strong_bear = htf_1w_bear and htf_1d_bear
        # Mixed/neutral: HTF disagree
        htf_mixed = (htf_1w_bull and htf_1d_bear) or (htf_1w_bear and htf_1d_bull)
        
        # === VOLATILITY FILTER ===
        atr_ratio = atr[i] / np.nanmean(atr[max(0,i-100):i]) if i >= 100 else 1.0
        vol_normal = atr_ratio < 2.5  # Avoid entering during 2.5x normal vol
        
        # === ENTRY LOGIC (LOOSE - OR logic, not AND) ===
        desired_signal = 0.0
        
        # STRONG TREND LONG: Both HTF bull + (Fisher cross OR RSI recovery OR HMA bull)
        if htf_strong_bull and vol_normal:
            if fisher_cross_up:
                # Fisher reversal from oversold
                desired_signal = SIZE_STRONG
            elif rsi_extreme_oversold and rsi_rising:
                # RSI oversold + starting to rise
                desired_signal = SIZE_BASE
            elif rsi[i] > 50.0 and rsi[i-1] <= 50.0:
                # RSI crossing above 50 = momentum shift
                desired_signal = SIZE_BASE
            elif hma_bull and above_sma50 and rsi[i] > 45.0:
                # HMA trend + above SMA50 + RSI neutral
                desired_signal = SIZE_BASE * 0.8
        
        # STRONG TREND SHORT: Both HTF bear + (Fisher cross OR RSI weakness OR HMA bear)
        elif htf_strong_bear and vol_normal:
            if fisher_cross_down:
                # Fisher reversal from overbought
                desired_signal = -SIZE_STRONG
            elif rsi_extreme_overbought and rsi_falling:
                # RSI overbought + starting to fall
                desired_signal = -SIZE_BASE
            elif rsi[i] < 50.0 and rsi[i-1] >= 50.0:
                # RSI crossing below 50 = weakness
                desired_signal = -SIZE_BASE
            elif hma_bear and below_sma50 and rsi[i] < 55.0:
                # HMA trend + below SMA50 + RSI neutral
                desired_signal = -SIZE_BASE * 0.8
        
        # MIXED HTF REGIME: Use Fisher + RSI for mean reversion
        elif htf_mixed and vol_normal:
            if fisher_oversold and rsi_extreme_oversold and above_sma200:
                # Double oversold + above long-term SMA = long
                desired_signal = SIZE_BASE * 0.8
            elif fisher_overbought and rsi_extreme_overbought and below_sma200:
                # Double overbought + below long-term SMA = short
                desired_signal = -SIZE_BASE * 0.8
        
        # MEAN REVERSION LONG: RSI extreme (works in any HTF regime)
        if desired_signal == 0.0 and vol_normal:
            if rsi_extreme_oversold and above_sma200 and fisher[i] < -1.0:
                desired_signal = SIZE_BASE * 0.8
            elif rsi_oversold and above_sma50 and rsi_rising:
                desired_signal = SIZE_BASE * 0.6
        
        # MEAN REVERSION SHORT: RSI extreme (works in any HTF regime)
        if desired_signal == 0.0 and vol_normal:
            if rsi_extreme_overbought and below_sma200 and fisher[i] > 1.0:
                desired_signal = -SIZE_BASE * 0.8
            elif rsi_overbought and below_sma50 and rsi_falling:
                desired_signal = -SIZE_BASE * 0.6
        
        # FISHER REVERSAL: Pure Fisher signals (work in any regime)
        if desired_signal == 0.0 and vol_normal:
            if fisher_cross_up and rsi[i] < 50.0:
                desired_signal = SIZE_BASE * 0.6
            elif fisher_cross_down and rsi[i] > 50.0:
                desired_signal = -SIZE_BASE * 0.6
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            # Update highest since entry for trailing
            highest_since_entry = max(highest_since_entry, high[i])
            # Check stoploss
            if low[i] < stop_price:
                stoploss_triggered = True
            # Trail stop: move up as price rises
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            # Update lowest since entry for trailing
            lowest_since_entry = min(lowest_since_entry, low[i])
            # Check stoploss
            if high[i] > stop_price:
                stoploss_triggered = True
            # Trail stop: move down as price falls
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
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.8
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                # Set stoploss
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