#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter
Hypothesis: On 12h timeframe, price breaking Camarilla R1/S1 levels from prior 1d bar with 1d EMA34 trend alignment, volume spike confirmation, and choppiness regime filter provides robust breakout signals. Uses discrete sizing (0.0, ±0.25) and ATR-based stoploss to control risk. Targets ~12-25 trades/year (~50-100 over 4 years) to stay within optimal trade frequency for 12h timeframe. Designed to work in both bull and bear markets via trend and regime filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss on 12h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume ratio (current / 20-period average) for spike confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)  # avoid division by zero
    
    # Calculate Camarilla levels from previous 1d bar (using 1d high/low/close)
    prev_high_1d = np.concatenate([[np.nan], df_1d['high'].values[:-1]])
    prev_low_1d = np.concatenate([[np.nan], df_1d['low'].values[:-1]])
    prev_close_1d = np.concatenate([[np.nan], df_1d['close'].values[:-1]])
    
    camarilla_r1 = prev_close_1d + ((prev_high_1d - prev_low_1d) * 1.1 / 12)
    camarilla_s1 = prev_close_1d - ((prev_high_1d - prev_low_1d) * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate Choppiness Index (CHOP) on 12h for regime filter
    def calculate_chop(high, low, close, period=14):
        """Calculate Choppiness Index: higher = more choppy, lower = more trending"""
        atr_sum = np.zeros_like(close)
        true_range = np.zeros_like(close)
        for i in range(1, len(close)):
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            true_range[i] = tr
            if i >= period:
                atr_sum[i] = atr_sum[i-1] + tr - (true_range[i-period+1] if i-period+1 >= 0 else 0)
            else:
                atr_sum[i] = atr_sum[i-1] + tr
        atr_period = np.full_like(close, np.nan)
        atr_period[period-1:] = atr_sum[period-1:] / period
        high_low = np.zeros_like(close)
        for i in range(len(close)):
            if i >= period:
                highest_high = np.max(high[i-period+1:i+1])
                lowest_low = np.min(low[i-period+1:i+1])
                high_low[i] = highest_high - lowest_low
            else:
                high_low[i] = np.nan
        chop = np.full_like(close, np.nan)
        mask = (high_low > 0) & ~np.isnan(atr_period)
        chop[mask] = 100 * np.log10(atr_period[mask] / high_low[mask]) / np.log10(period)
        return chop
    
    chop = calculate_chop(high, low, close, period=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1d EMA(34), ATR(14), volume MA(20), CHOP(14)
    start_idx = max(34, 14, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ratio[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_confirmed = vol_ratio[i] > 2.0  # volume at least 2.0x average
        trend_1d_up = close_val > ema_34_1d_aligned[i]
        trend_1d_down = close_val < ema_34_1d_aligned[i]
        chop_filter = chop[i] < 61.8  # only trade when not too choppy (trending regime)
        
        if position == 0:
            # Long: price breaks above Camarilla R1 AND 1d trend up AND volume confirmation AND chop filter
            long_signal = (close_val > camarilla_r1_aligned[i]) and trend_1d_up and vol_confirmed and chop_filter
            
            # Short: price breaks below Camarilla S1 AND 1d trend down AND volume confirmation AND chop filter
            short_signal = (close_val < camarilla_s1_aligned[i]) and trend_1d_down and vol_confirmed and chop_filter
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: trend flips down OR price hits ATR stoploss OR chop becomes too high
            if (not trend_1d_up) or (close_val < entry_price - 2.0 * atr[i]) or (chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend flips up OR price hits ATR stoploss OR chop becomes too high
            if (not trend_1d_down) or (close_val > entry_price + 2.0 * atr[i]) or (chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0