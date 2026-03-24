#!/usr/bin/env python3
"""
Experiment #884: 12h Primary + 1d/1w HTF — Fisher Transform + BB Compression + HTF Trend

Hypothesis: Bear/range markets (2025+) favor mean-reversion strategies with volatility
compression detection. Ehlers Fisher Transform excels at catching reversals in bear
rallies. Bollinger Band Width percentile identifies compression before expansion.
12h timeframe provides optimal trade frequency (20-50/year) with high signal quality.

Key innovations:
1. Ehlers Fisher Transform (period=9) - proven reversal detector in bear markets
2. BB Width percentile (100-period) - only trade when vol compressed <40th percentile
3. 1d HMA(21) for HTF trend bias - directional filter
4. 1w HMA(50) for major regime - avoid counter-trend in strong weekly trends
5. Volume confirmation - current vol > 0.8 * vol_ma(20)
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete sizing: 0.0, ±0.25, ±0.30

Entry conditions (LOOSE to ensure ≥10 trades/train, ≥3/test):
- LONG: Fisher crosses above -1.2 + BBW < 40th %ile + 1d HMA bull + volume confirm
- SHORT: Fisher crosses below +1.2 + BBW < 40th %ile + 1d HMA bear + volume confirm

Target: Sharpe>0.45, trades>=10 train, trades>=3 test, DD>-40%
Timeframe: 12h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_bbw_compression_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Identifies turning points by normalizing price and applying inverse tanh
    
    Steps:
    1. Calculate typical price: (high + low) / 2
    2. Smooth with EMA
    3. Normalize to -1 to +1 range using highest high / lowest low over period
    4. Apply Fisher: 0.5 * ln((1+x)/(1-x))
    5. Smooth Fisher with EMA
    """
    n = len(close)
    if n < period + 10:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Typical price
    typical = (high + low) / 2.0
    
    # Smooth typical price with EMA
    typical_smooth = pd.Series(typical).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Normalize to -1 to +1
    normalized = np.zeros(n)
    normalized[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(typical_smooth[i - period + 1:i + 1])
        lowest = np.min(typical_smooth[i - period + 1:i + 1])
        range_val = highest - lowest
        
        if range_val > 1e-10:
            normalized[i] = 0.999 * (2.0 * (typical_smooth[i] - lowest) / range_val - 1.0)
            # Clamp to avoid division by zero in Fisher
            normalized[i] = np.clip(normalized[i], -0.999, 0.999)
        else:
            normalized[i] = 0.0
    
    # Fisher Transform: 0.5 * ln((1+x)/(1-x))
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    for i in range(period, n):
        if not np.isnan(normalized[i]):
            x = normalized[i]
            if abs(x) < 0.999:
                fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
            else:
                fisher[i] = np.sign(x) * 3.0  # Cap at extreme
        else:
            fisher[i] = np.nan
    
    # Smooth Fisher with EMA
    fisher_smooth = pd.Series(fisher).ewm(span=3, min_periods=3, adjust=False).mean().values
    
    return fisher_smooth, normalized

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands with width calculation"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    # Middle band (SMA)
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    
    # Standard deviation
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    # Upper and lower bands
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    
    # Band width (normalized)
    bbw = np.zeros(n)
    bbw[:] = np.nan
    for i in range(period, n):
        if middle[i] > 1e-10:
            bbw[i] = (upper[i] - lower[i]) / middle[i]
        else:
            bbw[i] = 0.0
    
    return upper, lower, middle, bbw

def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate percentile rank of current BBW vs lookback period"""
    n = len(bbw)
    percentile = np.zeros(n)
    percentile[:] = np.nan
    
    for i in range(lookback, n):
        if not np.isnan(bbw[i]):
            window = bbw[i - lookback + 1:i + 1]
            valid_window = window[~np.isnan(window)]
            if len(valid_window) > 0:
                count_below = np.sum(valid_window[:-1] < bbw[i])
                percentile[i] = count_below / (len(valid_window) - 1) * 100.0
            else:
                percentile[i] = 50.0
        else:
            percentile[i] = np.nan
    
    return percentile

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.zeros(n)
    diff[:] = np.nan
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_n)
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

def calculate_volume_ma(volume, period=20):
    """Volume moving average"""
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
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 12h indicators
    fisher, fisher_normalized = calculate_fisher_transform(high, low, close, period=9)
    bb_upper, bb_lower, bb_middle, bbw = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    bbw_percentile = calculate_bbw_percentile(bbw, lookback=100)
    atr_14 = calculate_atr(high, low, close, period=14)
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
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(bbw_percentile[i]) or np.isnan(vol_ma_20[i]):
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
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === WEEKLY REGIME FILTER (1w HMA) ===
        # Avoid strong counter-trend trades in major weekly trends
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_long = False
        fisher_cross_short = False
        
        if i > 0 and not np.isnan(fisher[i-1]):
            # Long: Fisher crosses above -1.2 from below
            fisher_cross_long = (fisher[i-1] < -1.2) and (fisher[i] >= -1.2)
            # Short: Fisher crosses below +1.2 from above
            fisher_cross_short = (fisher[i-1] > 1.2) and (fisher[i] <= 1.2)
        
        # === BB WIDTH COMPRESSION ===
        # Only trade when volatility is compressed (bottom 40% of recent range)
        bbw_compressed = bbw_percentile[i] < 40.0
        
        # === VOLUME CONFIRMATION ===
        # Current volume should be at least 80% of 20-period MA
        volume_confirm = volume[i] >= 0.8 * vol_ma_20[i] if vol_ma_20[i] > 1e-10 else True
        
        # === ENTRY LOGIC (LOOSE FOR TRADES) ===
        desired_signal = 0.0
        
        # Long entry: Fisher cross + BBW compression + HTF bull + volume
        if htf_1d_bull and bbw_compressed and volume_confirm:
            if fisher_cross_long:
                desired_signal = SIZE_STRONG
            elif fisher[i] < -1.0 and htf_1w_bull:
                # Additional long signal if Fisher very oversold + weekly bull
                desired_signal = SIZE_BASE
        
        # Short entry: Fisher cross + BBW compression + HTF bear + volume
        elif htf_1d_bear and bbw_compressed and volume_confirm:
            if fisher_cross_short:
                desired_signal = -SIZE_STRONG
            elif fisher[i] > 1.0 and htf_1w_bear:
                # Additional short signal if Fisher very overbought + weekly bear
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