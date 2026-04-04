#!/usr/bin/env python3
"""
Experiment #3970: 1d Donchian(20) breakout + 1w HMA21 trend + volume confirmation
HYPOTHESIS: Daily Donchian breakouts aligned with weekly HMA21 trend capture multi-month swings with very low frequency. Volume > 1.5x MA(20) confirms breakout strength. ATR(14) trailing stop (2.0x) manages risk. Discrete sizing (0.25) minimizes fee drag. Target: 30-100 trades over 4 years (7-25/year). Works in bull/bear via weekly HMA21 regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3970_1d_donchian20_1w_hma21_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w HMA21 for trend regime ===
    df_1w = get_htf_data(prices, '1w')
    hma_period = 21
    half_period = hma_period // 2
    sqrt_period = int(np.sqrt(hma_period))
    
    # Calculate HMA: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    def wma(values, period):
        if period <= 0:
            return np.full_like(values, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # Vectorized WMA using pandas for simplicity and NaN handling
    close_series = pd.Series(close)
    wma_half = close_series.rolling(window=half_period, min_periods=half_period).apply(
        lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True
    ).values
    wma_full = close_series.rolling(window=hma_period, min_periods=hma_period).apply(
        lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True
    ).values
    wma_diff = 2 * wma_half - wma_full
    wma_diff_series = pd.Series(wma_diff)
    hma_values = wma_diff_series.rolling(window=sqrt_period, min_periods=sqrt_period).apply(
        lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True
    ).values
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_values)
    
    # === 1d Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 1d Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1d Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20, hma_period)  # DC lookback, vol MA, HMA period
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(hma_1w_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below Donchian lower band (trend reversal)
                elif price < lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above Donchian upper band (trend reversal)
                elif price > highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average) to filter noise
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Determine trend: bullish if price above 1w HMA21, bearish if below
            bullish = price > hma_1w_aligned[i]
            bearish = price < hma_1w_aligned[i]
            
            # Long entry: breakout above Donchian upper band in bullish regime
            long_breakout = price > highest_high[i-1] and bullish
            # Short entry: breakdown below Donchian lower band in bearish regime
            short_breakout = price < lowest_low[i-1] and bearish
            
            if long_breakout and not short_breakout:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_breakout and not long_breakout:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals