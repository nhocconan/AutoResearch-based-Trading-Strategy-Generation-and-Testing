#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian breakout with volume confirmation and chop regime filter
# Donchian(20) breakout provides clear entry/exit signals
# Volume confirmation ensures breakouts have conviction
# Choppiness index regime filter avoids false breakouts in sideways markets
# Designed to work in both bull and bear markets by filtering trades with regime
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag

name = "12h_1d_donchian_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    def rolling_max(arr, window):
        res = np.full(len(arr), np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full(len(arr), np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    upper_20 = rolling_max(high_1d, 20)
    lower_20 = rolling_min(low_1d, 20)
    
    # Calculate 1d ATR(14) for stoploss and volume normalization
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # Calculate 1d average volume (20-period) for volume confirmation
    def sma(values, period):
        res = np.full(len(values), np.nan)
        for i in range(period-1, len(values)):
            res[i] = np.mean(values[i-period+1:i+1])
        return res
    
    avg_volume_20 = sma(volume_1d, 20)
    
    # Calculate 1d Choppiness Index (14-period) for regime filter
    def choppiness_index(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high[1:] - low[:-1])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Sum of TR over period
        tr_sum = np.full(len(tr), np.nan)
        for i in range(period-1, len(tr)):
            tr_sum[i] = np.sum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        hh = rolling_max(high, period)
        ll = rolling_min(low, period)
        
        # Chop = 100 * log10(sum(tr) / (hh - ll)) / log10(period)
        denom = hh - ll
        chop = np.full(len(tr), np.nan)
        mask = (denom > 0) & (~np.isnan(tr_sum))
        chop[mask] = 100 * np.log10(tr_sum[mask] / denom[mask]) / np.log10(period)
        return chop
    
    chop_1d = choppiness_index(high_1d, low_1d, close_1d, 14)
    
    # Align 1d indicators to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    avg_volume_20_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_20)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(avg_volume_20_aligned[i]) or
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when market is trending (CHOP < 38.2) or extreme ranging (CHOP > 61.8)
        # Avoid middling chop values (38.2-61.8) where breakouts often fail
        chop_val = chop_1d_aligned[i]
        if 38.2 <= chop_val <= 61.8:
            # Choppy market - reduce position size or stay flat
            if position == 1:
                # Exit long if price returns to mid-channel
                mid_channel = (upper_20_aligned[i] + lower_20_aligned[i]) / 2
                if close[i] >= mid_channel:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.15  # Reduced size in chop
            elif position == -1:
                # Exit short if price returns to mid-channel
                mid_channel = (upper_20_aligned[i] + lower_20_aligned[i]) / 2
                if close[i] <= mid_channel:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.15  # Reduced size in chop
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_20_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions: stoploss or Donchian lower band break
            if close[i] < lower_20_aligned[i] - 0.5 * atr_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif not volume_confirmed and close[i] < upper_20_aligned[i]:
                # Weak volume, take profit on retracement
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: stoploss or Donchian upper band break
            if close[i] > upper_20_aligned[i] + 0.5 * atr_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif not volume_confirmed and close[i] > lower_20_aligned[i]:
                # Weak volume, take profit on retracement
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions: Donchian breakout with volume confirmation
            if volume_confirmed:
                if close[i] > upper_20_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < lower_20_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals