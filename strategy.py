#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d HMA trend filter and volume confirmation
# - Long: Price breaks above 20-period Donchian high + 1d HMA(21) rising + 1d volume > 1.5x 20-period average volume
# - Short: Price breaks below 20-period Donchian low + 1d HMA(21) falling + same volume confirmation
# - Exit: Close-based reversal - exit long when price < Donchian low, exit short when price > Donchian high
# - Stoploss: ATR-based - exit when price moves against position by 2.0 * ATR(14) on 4h
# - Position sizing: 0.25 (discrete level)
# - Uses HMA on 1d for trend filter to avoid counter-trend trades, volume confirmation for institutional participation
# - Target: 75-200 total trades over 4 years (19-50/year) to stay within HARD MAX: 400 total

name = "4h_1d_donchian_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def hull_moving_average(data, period):
    """Calculate Hull Moving Average"""
    if len(data) < period:
        return np.full_like(data, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA calculation
    def wma(arr, window):
        if len(arr) < window:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(arr, weights/weights.sum(), mode='valid')
    
    # Handle edge cases
    if len(data) < period:
        return np.full_like(data, np.nan)
    
    # Calculate WMAs
    wma_half = wma(data, half_period)
    wma_full = wma(data, period)
    
    # Align arrays (WMA reduces length)
    if len(wma_half) < half_period or len(wma_full) < period:
        return np.full_like(data, np.nan)
    
    # Calculate raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half[-len(wma_full):] - wma_full
    
    # Final WMA of raw HMA with sqrt_period
    if len(raw_hma) < sqrt_period:
        return np.full_like(data, np.nan)
    hma_result = wma(raw_hma, sqrt_period)
    
    # Pad to original length
    result = np.full_like(data, np.nan)
    start_idx = period - half_period - sqrt_period + 1
    end_idx = start_idx + len(hma_result)
    if start_idx >= 0 and end_idx <= len(data):
        result[start_idx:end_idx] = hma_result
    
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 30:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Pre-compute 1d data
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian channel (20-period) for 4h
    high_series = pd.Series(high_4h)
    low_series = pd.Series(low_4h)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d HMA(21) for trend filter
    hma_21_1d = hull_moving_average(close_1d, 21)
    hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate 1d HMA slope (rising/falling)
    hma_slope = np.diff(hma_21_aligned, prepend=np.nan)
    hma_rising = hma_slope > 0
    hma_falling = hma_slope < 0
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 4h ATR (14-period) for stoploss
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First TR is 0 (no previous close)
    
    # Wilder's ATR smoothing
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.zeros_like(data)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])  # First value is simple average
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_14_4h = wilders_smoothing(tr, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(20, n):  # Start after warmup period (need at least 20 for Donchian)
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(atr_14_4h[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h close
        close_price = close_4h[i]
        
        # Get current 1d volume for confirmation
        volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        volume_confirmation = volume_1d_current > 1.5 * volume_ma_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high + HMA rising + volume confirmation
            if (close_price > donchian_high[i] and hma_rising[i] and volume_confirmation):
                position = 1
                entry_price = close_price
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low + HMA falling + volume confirmation
            elif (close_price < donchian_low[i] and hma_falling[i] and volume_confirmation):
                position = -1
                entry_price = close_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or stoploss
            # Calculate stoploss level
            if position == 1:  # Long position
                stop_loss = entry_price - 2.0 * atr_14_4h[i]
                # Exit conditions: price < Donchian low OR stoploss hit
                if close_price < donchian_low[i] or close_price <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                stop_loss = entry_price + 2.0 * atr_14_4h[i]
                # Exit conditions: price > Donchian high OR stoploss hit
                if close_price > donchian_high[i] or close_price >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals