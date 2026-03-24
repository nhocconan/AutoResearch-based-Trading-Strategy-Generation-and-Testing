#!/usr/bin/env python3
"""
Experiment #1063: 6h Primary + 1d/1w HTF — Fisher Transform + HMA Trend + Volume Confirm

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). Using Ehlers Fisher Transform
for precise reversal entries, combined with 1w/1d HMA trend bias and volume confirmation, will
capture multi-day swings while avoiding whipsaws. Fisher Transform normalizes price to Gaussian
distribution, making extreme values (-2 to +2) reliable reversal signals.

Key innovations:
1. Ehlers Fisher Transform (period=9): Converts price to bounded Gaussian (-2 to +2)
   - Long when Fisher crosses above -1.2 from below (oversold reversal)
   - Short when Fisher crosses below +1.2 from above (overbought reversal)
2. 1w HMA(21) for long-term bias: Only long if price > 1w_HMA, only short if price < 1w_HMA
3. 1d HMA(21) for intermediate confirmation: Strengthens signal when aligned with 1w
4. Volume spike filter: Entry volume > 1.3x 20-bar avg volume (confirms institutional interest)
5. 6h ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this should work on 6h:
- 6h captures 2-4 day swings (perfect for Fisher reversal signals)
- Fisher Transform excels at catching reversals in bear/range markets (2022-2025)
- 1w/1d HMA filter prevents counter-trend trades (major improvement over pure Fisher)
- Volume confirmation reduces false signals (critical for lower TF)
- Lenient Fisher thresholds (-1.2/+1.2) ensure 30-60 trades/year target is met

Entry conditions (LOOSE to guarantee trades on ALL symbols):
- LONG: price>1w_HMA + Fisher crosses above -1.2 + volume>1.3x_avg
- SHORT: price<1w_HMA + Fisher crosses below +1.2 + volume>1.3x_avg
- Strong signal when 1d_HMA also aligned (SIZE_STRONG vs SIZE_BASE)

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_hma_volume_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - converts price to bounded Gaussian distribution
    Formula from "Cybernetic Analysis for Stocks and Futures" by John Ehlers
    
    Steps:
    1. Calculate median price: (high + low) / 2
    2. Normalize: (median - lowest_low) / (highest_high - lowest_low)
    3. Scale to -1 to +1: 2 * normalized - 1 (with epsilon to avoid division by zero)
    4. Fisher: 0.5 * ln((1 + scaled) / (1 - scaled))
    5. Signal line: previous Fisher value
    
    Output bounded between -2 and +2, extreme values indicate reversals
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    median = (high + low) / 2.0
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_signal = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range < 1e-10:
            continue
        
        # Normalize to 0-1
        normalized = (median[i] - lowest_low) / price_range
        
        # Scale to -0.99 to +0.99 (avoid division by zero in log)
        scaled = max(-0.99, min(0.99, 2.0 * normalized - 1.0))
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + scaled) / (1.0 - scaled))
        
        # Signal line is previous Fisher value
        if i > period and not np.isnan(fisher[i-1]):
            fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

def calculate_volume_ma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    vol_ma_20 = calculate_volume_ma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
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
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 1e-10:
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
        
        # === HTF BIAS (1w HMA for long-term direction) ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # 1d HMA alignment strengthens signal
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_spike = volume[i] > 1.3 * vol_ma_20[i]
        
        # === FISHER TRANSFORM CROSSOVER DETECTION ===
        # Long: Fisher crosses above -1.2 from below
        fisher_cross_long = (fisher[i] > -1.2) and (fisher_signal[i] <= -1.2)
        # Short: Fisher crosses below +1.2 from above
        fisher_cross_short = (fisher[i] < 1.2) and (fisher_signal[i] >= 1.2)
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG entry: 1w bias up + Fisher reversal + volume confirm
        if price_above_1w and fisher_cross_long and volume_spike:
            # Stronger if 1d also aligned
            if price_above_1d:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT entry: 1w bias down + Fisher reversal + volume confirm
        elif price_below_1w and fisher_cross_short and volume_spike:
            # Stronger if 1d also aligned
            if price_below_1d:
                desired_signal = -SIZE_STRONG
            else:
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