#!/usr/bin/env python3
"""
Experiment #783: 6h Primary + 1d/1w HTF — Bollinger Mean Reversion with Triple HTF Filter

Hypothesis: 6h timeframe sits between 4h (too noisy) and 12h (too slow). 
Using Bollinger Band %B for mean reversion entries + 1w HMA for secular trend 
+ 1d RSI for momentum confirmation should capture reversals in bear/range markets
while avoiding counter-trend trades in strong trends.

Key innovations:
1. 1w HMA(21) for secular bias (bull/bear market regime)
2. 1d RSI(14) for intermediate momentum filter
3. 6h Bollinger %B for precise entry timing (mean reversion)
4. 6h ATR(14) for volatility filter and 2.5x trailing stops
5. BB Width percentile for regime detection (squeeze = breakout soon)
6. Asymmetric sizing: 0.30 on strong confluence, 0.20 on weak

Entry conditions (LOOSE to ensure ≥30 trades/train, ≥3/test):
- LONG: 1w HMA bull + 1d RSI<50 + 6h BB %B < 0.15 (oversold)
- SHORT: 1w HMA bear + 1d RSI>50 + 6h BB %B > 0.85 (overbought)
- Squeeze breakout: BB Width < 20th percentile + price breaks BB

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_bb_rsi_triple_htf_1w1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
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
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands with %B indicator"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    # %B indicator: where price is relative to bands (0=lower, 1=upper)
    band_width = upper - lower
    pct_b = np.zeros(n)
    pct_b[:] = np.nan
    mask = band_width > 1e-10
    pct_b[mask] = (close[mask] - lower[mask]) / band_width[mask]
    
    # BB Width as % of SMA (for squeeze detection)
    bb_width_pct = np.zeros(n)
    bb_width_pct[:] = np.nan
    mask2 = sma > 1e-10
    bb_width_pct[mask2] = (band_width[mask2] / sma[mask2]) * 100.0
    
    return upper, lower, pct_b, bb_width_pct

def calculate_bb_width_percentile(bb_width, lookback=100):
    """Calculate rolling percentile of BB Width for squeeze detection"""
    n = len(bb_width)
    percentile = np.zeros(n)
    percentile[:] = np.nan
    
    for i in range(lookback, n):
        if not np.isnan(bb_width[i]):
            window = bb_width[i-lookback+1:i+1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                percentile[i] = np.sum(valid <= bb_width[i]) / len(valid) * 100.0
    
    return percentile

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    rsi_1d_raw = calculate_rsi(df_1d['close'].values, period=14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_raw)
    
    # Calculate 6h indicators
    hma_6h_16 = calculate_hma(close, period=16)
    hma_6h_48 = calculate_hma(close, period=48)
    atr_14 = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, pct_b, bb_width_pct = calculate_bollinger(close, period=20, std_mult=2.0)
    bb_width_pctile = calculate_bb_width_percentile(bb_width_pct, lookback=100)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_6h_16[i]) or np.isnan(hma_6h_48[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(pct_b[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(rsi_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SECULAR TREND (1w HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === INTERMEDIATE MOMENTUM (1d RSI) ===
        rsi_1d_neutral = 40.0 < rsi_1d_aligned[i] < 60.0
        rsi_1d_bull = rsi_1d_aligned[i] < 50.0  # Room to go up
        rsi_1d_bear = rsi_1d_aligned[i] > 50.0  # Room to go down
        
        # === 6h MEAN REVERSION (Bollinger %B) ===
        bb_oversold = pct_b[i] < 0.15  # Near lower band
        bb_overbought = pct_b[i] > 0.85  # Near upper band
        bb_extreme_oversold = pct_b[i] < 0.05  # Very oversold
        bb_extreme_overbought = pct_b[i] > 0.95  # Very overbought
        
        # === 6h HMA TREND CONFIRMATION ===
        hma_6h_bull = hma_6h_16[i] > hma_6h_48[i]
        hma_6h_bear = hma_6h_16[i] < hma_6h_48[i]
        
        # === BB SQUEEZE DETECTION ===
        bb_squeeze = not np.isnan(bb_width_pctile[i]) and bb_width_pctile[i] < 20.0
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        
        # LONG: 1w bull + 1d RSI has room + 6h BB oversold
        if htf_1w_bull:
            if bb_oversold and rsi_1d_bull:
                if bb_extreme_oversold or (bb_squeeze and hma_6h_bull):
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            # Also allow HMA crossover entries in bull market
            elif i > 0 and not np.isnan(hma_6h_16[i-1]) and not np.isnan(hma_6h_48[i-1]):
                if (hma_6h_16[i-1] <= hma_6h_48[i-1]) and (hma_6h_16[i] > hma_6h_48[i]):
                    desired_signal = SIZE_BASE
        
        # SHORT: 1w bear + 1d RSI has room + 6h BB overbought
        elif htf_1w_bear:
            if bb_overbought and rsi_1d_bear:
                if bb_extreme_overbought or (bb_squeeze and hma_6h_bear):
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            # Also allow HMA crossover entries in bear market
            elif i > 0 and not np.isnan(hma_6h_16[i-1]) and not np.isnan(hma_6h_48[i-1]):
                if (hma_6h_16[i-1] >= hma_6h_48[i-1]) and (hma_6h_16[i] < hma_6h_48[i]):
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
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
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
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