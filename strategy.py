#!/usr/bin/env python3
"""
Experiment #311: 6h Primary + 1d/1w HTF — Ehlers Fisher Transform + Choppiness Regime v1

Hypothesis: Ehlers Fisher Transform excels at catching reversals in bear/range markets (2025 test).
Combined with Choppiness Index regime detection and 1d/1w HMA for trend bias.

Why 6h is different from 4h/12h:
- 6h captures 4 candles per day vs 4h's 6 or 12h's 2
- Better balance between noise reduction and signal frequency
- Target: 30-60 trades/year (similar to 4h but cleaner signals)

Key features:
1. EHLERS FISHER TRANSFORM: period=9, long when Fisher crosses above -1.5, short when crosses below +1.5
   This catches reversals in bear rallies better than RSI
2. CHOPPINESS INDEX regime: CHOP > 55 = range (Fisher mean reversion), CHOP < 45 = trending (HMA breakout)
3. 1d/1w HMA alignment: Only take longs when price > 1d HMA, shorts when price < 1d HMA
4. ASYMMETRIC SIZING: 0.30 when 1d+1w aligned, 0.20 otherwise (discrete levels)
5. STOPLOSS: 2.5x ATR from entry (fixed, not trailing - more reliable)

Entry Logic:
- Range regime (CHOP > 55): Fisher < -1.5 + price > 1d HMA → long; Fisher > +1.5 + price < 1d HMA → short
- Trend regime (CHOP < 45): Price breaks 6h HMA + 1d HMA aligned → follow trend
- 1w HMA for major trend boost (increases size when aligned)

Position sizing: 0.20 base, 0.30 when HTF aligned (discrete: 0.0, ±0.20, ±0.30)
Stoploss: 2.5x ATR from entry price

Target: Sharpe>0.50, DD>-35%, trades>=30 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_chop_regime_hma_1d1w_v1"
timeframe = "6h"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = choppy/range bound
    CHOP < 38.2 = trending
    Using 55/45 thresholds for 6h timeframe
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Highlights turning points in price action
    
    Formula:
    1. Calculate typical price: (High + Low) / 2
    2. Normalize: (Price - Lowest) / (Highest - Lowest)
    3. Transform: 0.5 * ln((1 + X) / (1 - X)) where X = 2*normalized - 1
    4. Smooth with EMA
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Typical price
    typical = (high + low) / 2.0
    
    # Normalize price over lookback period
    normalized = np.zeros(n)
    normalized[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(typical[i-period+1:i+1])
        lowest = np.min(typical[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range > 1e-10:
            normalized[i] = (typical[i] - lowest) / price_range
        else:
            normalized[i] = 0.5
    
    # Clamp to avoid division by zero in log
    normalized = np.clip(normalized, 0.001, 0.999)
    
    # Fisher transform
    fisher_raw = np.zeros(n)
    fisher_raw[:] = np.nan
    
    for i in range(period, n):
        if not np.isnan(normalized[i]):
            x = 2.0 * normalized[i] - 1.0
            x = np.clip(x, -0.999, 0.999)  # Safety clamp
            fisher_raw[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
    
    # Smooth Fisher with EMA
    fisher = pd.Series(fisher_raw).ewm(span=3, min_periods=3, adjust=False).mean().values
    
    # Fisher trigger (previous bar's fisher for signal generation)
    fisher_trigger = np.zeros(n)
    fisher_trigger[:] = np.nan
    for i in range(1, n):
        fisher_trigger[i] = fisher[i-1]
    
    return fisher, fisher_trigger

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    hma_6h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, period=9)
    sma_200 = calculate_sma(close, 200)
    rsi = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Regime memory for hysteresis
    prev_regime = 0  # 0=unknown, 1=trending, 2=range
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_6h[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION with HYSTERESIS ===
        range_threshold = 55.0
        trend_threshold = 45.0
        
        if chop[i] > range_threshold:
            current_regime = 2  # range/choppy
        elif chop[i] < trend_threshold:
            current_regime = 1  # trending
        else:
            current_regime = prev_regime  # use memory
        
        prev_regime = current_regime
        
        # === HTF BIAS ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # 1w for major trend (optional boost)
        htf_1w_bull = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        htf_1w_bear = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crosses above -1.5 from below = long signal
        # Fisher crosses below +1.5 from above = short signal
        fisher_long_cross = False
        fisher_short_cross = False
        
        if not np.isnan(fisher[i]) and not np.isnan(fisher_trigger[i]):
            # Long: Fisher was below -1.5, now crosses above
            if fisher_trigger[i] < -1.5 and fisher[i] >= -1.5:
                fisher_long_cross = True
            # Short: Fisher was above +1.5, now crosses below
            if fisher_trigger[i] > 1.5 and fisher[i] <= 1.5:
                fisher_short_cross = True
        
        # Fisher extreme levels (for range regime)
        fisher_extreme_low = not np.isnan(fisher[i]) and fisher[i] < -1.8
        fisher_extreme_high = not np.isnan(fisher[i]) and fisher[i] > 1.8
        
        # === RSI CONFIRMATION ===
        rsi_oversold = not np.isnan(rsi[i]) and rsi[i] < 35.0
        rsi_overbought = not np.isnan(rsi[i]) and rsi[i] > 65.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: RANGE/CHOPPY (Fisher mean reversion)
        if current_regime == 2:
            # Long: Fisher extreme low + above 1d HMA + RSI oversold
            if fisher_extreme_low and htf_1d_bull and rsi_oversold:
                desired_signal = SIZE_STRONG if htf_1w_bull else SIZE_BASE
            
            # Short: Fisher extreme high + below 1d HMA + RSI overbought
            elif fisher_extreme_high and htf_1d_bear and rsi_overbought:
                desired_signal = -SIZE_STRONG if htf_1w_bear else -SIZE_BASE
        
        # REGIME 2: TRENDING (Fisher cross + HMA confirmation)
        elif current_regime == 1:
            # Long: Fisher cross + 6h HMA bull + 1d HMA bull
            if fisher_long_cross and hma_bull and htf_1d_bull:
                desired_signal = SIZE_STRONG if htf_1w_bull else SIZE_BASE
            
            # Short: Fisher cross + 6h HMA bear + 1d HMA bear
            elif fisher_short_cross and hma_bear and htf_1d_bear:
                desired_signal = -SIZE_STRONG if htf_1w_bear else -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
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
        
        signals[i] = final_signal
    
    return signals