#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 1d Choppiness regime + volume confirmation
# Donchian provides clear breakout signals, Choppiness filter adapts to market regime (trending/range),
# Volume confirms institutional participation. Works in both bull and bear markets.
# Target: 20-40 trades/year (80-160 over 4 years) to avoid excessive fees.
name = "4h_DonchianBreakout_1dChop_Regime_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    dc_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dc_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate Choppiness Index on 1d (14-period)
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        atr = []
        for i in range(len(close_arr)):
            if i == 0:
                tr = high_arr[i] - low_arr[i]
            else:
                tr = max(high_arr[i] - low_arr[i], 
                         abs(high_arr[i] - close_arr[i-1]),
                         abs(low_arr[i] - close_arr[i-1]))
            atr.append(tr)
        
        # Smooth ATR with Wilder's smoothing (equivalent to RMA)
        atr_arr = np.array(atr)
        atr_ma = np.full_like(atr_arr, np.nan)
        if len(atr_arr) >= period:
            atr_ma[period-1] = np.mean(atr_arr[:period])
            for i in range(period, len(atr_arr)):
                atr_ma[i] = (atr_ma[i-1] * (period-1) + atr_arr[i]) / period
        
        # Calculate highest high and lowest low over period
        highest_high = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        
        # Chop = 100 * log10(sum(atr_ma/period) / (highest_high - lowest_low)) / log10(period)
        chop = np.full_like(close_arr, np.nan)
        for i in range(len(close_arr)):
            if not np.isnan(atr_ma[i]) and not np.isnan(highest_high[i]) and not np.isnan(lowest_low[i]):
                if highest_high[i] - lowest_low[i] > 0:
                    chop[i] = 100 * np.log10(atr_ma[i] * period / (highest_high[i] - lowest_low[i])) / np.log10(period)
        return chop
    
    chop_1d = calculate_chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    chop_1d_4h = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for Donchian and Chop
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(chop_1d_4h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.8 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 1.8
        
        # Regime filter: Chop > 61.8 = ranging (mean revert), Chop < 38.2 = trending (trend follow)
        chop_val = chop_1d_4h[i]
        is_ranging = chop_val > 61.8
        is_trending = chop_val < 38.2
        
        if position == 0:
            # Long: Breakout above upper band + trending OR ranging with volume
            if close[i] > dc_upper[i] and (is_trending or (is_ranging and vol_spike)):
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below lower band + trending OR ranging with volume
            elif close[i] < dc_lower[i] and (is_trending or (is_ranging and vol_spike)):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price touches opposite band OR chop becomes extremely high (choppy market)
            if close[i] < dc_lower[i] or chop_val > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price touches opposite band OR chop becomes extremely high
            if close[i] > dc_upper[i] or chop_val > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals