#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter
Hypothesis: On 4h timeframe, price breaking Camarilla R1/S1 levels with 1d EMA34 trend alignment and volume confirmation provides edge. Choppiness Index (CHOP) filter avoids whipsaws in ranging markets. In bull markets: 1d EMA34 uptrend + R1 breakout with volume = long setup. In bear markets: 1d EMA34 downtrend + S1 breakdown with volume = short setup. CHOP > 61.8 avoids range-bound false breakouts. Discrete sizing (0.0, ±0.25) minimizes fee churn. Targets ~25-35 trades/year (~100-140 over 4 years) to avoid fee drag on 4h timeframe. ATR-based stoploss (2.0x) manages risk.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss on 4h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume ratio (current / 20-period average) for spike confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)  # avoid division by zero
    
    # Calculate Choppiness Index (CHOP) on 4h for regime filter
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        # True Range
        tr1 = high_arr[1:] - low_arr[1:]
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Sum of TR over window
        tr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum().values
        
        # Highest high and lowest low over window
        hh = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        ll = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        
        # Chop = 100 * log10(tr_sum / (hh - ll)) / log10(window)
        # Avoid division by zero and log of zero
        hh_ll = hh - ll
        chop = np.zeros_like(close_arr)
        chop[:] = np.nan
        valid = (hh_ll > 0) & (~np.isnan(tr_sum))
        chop[valid] = 100 * np.log10(tr_sum[valid] / hh_ll[valid]) / np.log10(window)
        return chop
    
    chop = calculate_chop(high, low, close, window=14)
    
    # Calculate Camarilla levels from previous 4h bar
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    prev_close = np.concatenate([[np.nan], close[:-1]])
    
    camarilla_r1 = prev_close + ((prev_high - prev_low) * 1.1 / 6)
    camarilla_s1 = prev_close - ((prev_high - prev_low) * 1.1 / 6)
    
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
            np.isnan(chop[i]) or
            np.isnan(camarilla_r1[i]) or
            np.isnan(camarilla_s1[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_confirmed = vol_ratio[i] > 1.8  # volume at least 1.8x average
        chop_filter = chop[i] <= 61.8  # only trade when not too choppy (trending market)
        trend_1d_up = close_val > ema_34_1d_aligned[i]
        trend_1d_down = close_val < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1 AND 1d trend up AND volume confirmation AND chop filter
            long_signal = (close_val > camarilla_r1[i]) and trend_1d_up and vol_confirmed and chop_filter
            
            # Short: price breaks below Camarilla S1 AND 1d trend down AND volume confirmation AND chop filter
            short_signal = (close_val < camarilla_s1[i]) and trend_1d_down and vol_confirmed and chop_filter
            
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
            # Exit: trend flips down OR price hits ATR stoploss OR chop becomes too high (range)
            if (not trend_1d_up) or (close_val < entry_price - 2.0 * atr[i]) or (chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend flips up OR price hits ATR stoploss OR chop becomes too high (range)
            if (not trend_1d_down) or (close_val > entry_price + 2.0 * atr[i]) or (chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0