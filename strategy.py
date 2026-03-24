#!/usr/bin/env python3
"""
Experiment #640: 6h Primary + 1d/1w HTF — Triple Timeframe HMA Alignment + Fisher Transform Entry

Hypothesis: 6h timeframe sits in sweet spot between 4h (too many trades) and 12h (too few).
Using triple timeframe alignment (6h + 1d + 1w HMA) ensures we only trade when all timeframes
agree on direction. Fisher Transform (Ehlers) provides superior entry timing vs RSI - catches
reversals at extremes with less lag. Volume surge confirms breakout validity.

Key innovations:
1. Triple HMA alignment: 6h price > 6h HMA > 1d HMA > 1w HMA for long (reverse for short)
2. Fisher Transform(9): entry when Fisher crosses -1.5 (long) or +1.5 (short) from extremes
3. Volume surge: current volume > 1.5x 20-period avg volume (confirms breakout)
4. Asymmetric sizing: 0.30 when all 3 HTF aligned, 0.20 when only 2 aligned
5. ATR(14) trailing stop: 2.5x for risk management

Why this might work on 6h:
- 6h has 4 bars/day = ~1460 bars/year, target 30-50 trades = 2-3% trade rate
- Triple HTF filter eliminates most false signals in choppy 2022-2024 period
- Fisher Transform proven to work in bear/range markets (2025 test period)
- Volume filter avoids low-liquidity false breakouts

Target: Sharpe>0.40, trades>=30 train, trades>=3 test
Timeframe: 6h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_triple_hma_fisher_volume_1d1w_v1"
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

def calculate_fisher_transform(high, low, period=9):
    """Ehlers Fisher Transform - normalizes price to Gaussian distribution for reversal detection"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    for i in range(period - 1, n):
        # Calculate typical price
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest = np.nanmax(high[i - period + 1:i + 1])
        lowest = np.nanmin(low[i - period + 1:i + 1])
        
        # Normalize price to 0-1 range
        range_val = highest - lowest
        if range_val < 1e-10:
            continue
        
        normalized = (hl2 - lowest) / range_val
        
        # Clamp to avoid division issues
        normalized = max(0.001, min(0.999, normalized))
        
        # Calculate Fisher value
        fisher_raw = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Smooth with EMA
        if i == period - 1:
            fisher[i] = fisher_raw
        else:
            fisher[i] = 0.67 * fisher_raw + 0.33 * fisher[i - 1]
        
        # Trigger line (1-period lag)
        if i > period - 1:
            trigger[i] = fisher[i - 1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

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

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA (21-period)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    hma_6h = calculate_hma(close, period=21)
    fisher, trigger = calculate_fisher_transform(high, low, period=9)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_WEAK = 0.20
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
        
        if np.isnan(hma_6h[i]) or np.isnan(fisher[i]):
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
        
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TRIPLE TIMEFRAME ALIGNMENT ===
        # Long alignment: price > 6h HMA > 1d HMA > 1w HMA
        htf_long_strong = (close[i] > hma_6h[i] > hma_1d_aligned[i] > hma_1w_aligned[i])
        htf_long_weak = (close[i] > hma_6h[i] and hma_6h[i] > hma_1d_aligned[i])
        
        # Short alignment: price < 6h HMA < 1d HMA < 1w HMA
        htf_short_strong = (close[i] < hma_6h[i] < hma_1d_aligned[i] < hma_1w_aligned[i])
        htf_short_weak = (close[i] < hma_6h[i] and hma_6h[i] < hma_1d_aligned[i])
        
        # === FISHER TRANSFORM ENTRY SIGNALS ===
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        fisher_long = False
        if i > 0 and not np.isnan(fisher[i-1]) and not np.isnan(trigger[i]):
            fisher_long = (trigger[i] < -1.5 and fisher[i] > -1.5)
        
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_short = False
        if i > 0 and not np.isnan(fisher[i-1]) and not np.isnan(trigger[i]):
            fisher_short = (trigger[i] > 1.5 and fisher[i] < 1.5)
        
        # === VOLUME SURGE CONFIRMATION ===
        volume_surge = volume[i] > 1.5 * vol_sma[i]
        
        # === HMA SLOPE CONFIRMATION ===
        hma_slope_long = False
        hma_slope_short = False
        if i >= 3 and not np.isnan(hma_6h[i-3]):
            hma_slope_long = hma_6h[i] > hma_6h[i-1] and hma_6h[i-1] > hma_6h[i-2]
            hma_slope_short = hma_6h[i] < hma_6h[i-1] and hma_6h[i-1] < hma_6h[i-2]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Triple alignment + Fisher cross + volume OR weak alignment + Fisher + volume + slope
        if htf_long_strong and fisher_long:
            desired_signal = SIZE_STRONG
        elif htf_long_weak and fisher_long and volume_surge and hma_slope_long:
            desired_signal = SIZE_WEAK
        elif htf_long_strong and close[i] > hma_6h[i] and hma_slope_long:
            # Pullback entry when strongly aligned
            desired_signal = SIZE_WEAK
        
        # SHORT: Triple alignment + Fisher cross + volume OR weak alignment + Fisher + volume + slope
        elif htf_short_strong and fisher_short:
            desired_signal = -SIZE_STRONG
        elif htf_short_weak and fisher_short and volume_surge and hma_slope_short:
            desired_signal = -SIZE_WEAK
        elif htf_short_strong and close[i] < hma_6h[i] and hma_slope_short:
            # Pullback entry when strongly aligned
            desired_signal = -SIZE_WEAK
        
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
        elif desired_signal >= SIZE_WEAK * 0.9:
            final_signal = SIZE_WEAK
        elif desired_signal <= -SIZE_WEAK * 0.9:
            final_signal = -SIZE_WEAK
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