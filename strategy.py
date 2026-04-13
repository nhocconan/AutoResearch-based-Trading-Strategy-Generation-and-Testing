#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout from 1d with volume confirmation and chop regime filter.
    # Long when price breaks above Camarilla H3 with volume spike and chop < 61.8 (trending).
    # Short when price breaks below Camarilla L3 with volume spike and chop < 61.8.
    # Exit when price returns to Camarilla pivot point (mean reversion to equilibrium).
    # Uses discrete size 0.25 to minimize fee churn. Target: 50-150 trades over 4 years.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume mean (20-period) with min_periods
    volume_series = pd.Series(volume_1d)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (14-period) for regime filter
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        # Wilder's smoothing for ATR
        if len(tr) > period:
            atr[period] = np.nansum(tr[1:period+1])
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        # Sum of ATR over period
        sum_atr = np.zeros_like(close)
        for i in range(period, len(close)):
            sum_atr[i] = np.nansum(atr[i-period+1:i+1])
        # Max high - min low over period
        max_high = np.zeros_like(high)
        min_low = np.zeros_like(low)
        for i in range(period-1, len(high)):
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        # Chop = 100 * log10(sum(ATR) / (maxH - minL)) / log10(period)
        range_hl = max_high - min_low
        chop = np.full_like(close, 50.0)  # default to neutral
        for i in range(period, len(close)):
            if range_hl[i] > 0:
                chop[i] = 100 * np.log10(sum_atr[i] / range_hl[i]) / np.log10(period)
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    def calculate_camarilla(high, low, close):
        # Camarilla levels calculated from previous day's OHLC
        pivot = (high + low + close) / 3.0
        range_hl = high - low
        # Resistance levels
        R4 = close + range_hl * 1.5000
        R3 = close + range_hl * 1.2500
        R2 = close + range_hl * 1.1666
        R1 = close + range_hl * 1.0833
        # Support levels
        S1 = close - range_hl * 1.0833
        S2 = close - range_hl * 1.1666
        S3 = close - range_hl * 1.2500
        S4 = close - range_hl * 1.5000
        return pivot, R1, R2, R3, R4, S1, S2, S3, S4
    
    # Calculate Camarilla for each 1d bar (using previous day's data)
    camarilla_pivot = np.full_like(close_1d, np.nan)
    camarilla_R3 = np.full_like(close_1d, np.nan)
    camarilla_L3 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        pivot, R1, R2, R3, R4, S1, S2, S3, S4 = calculate_camarilla(
            high_1d[i-1], low_1d[i-1], close_1d[i-1]
        )
        camarilla_pivot[i] = pivot
        camarilla_R3[i] = R3
        camarilla_L3[i] = S3
    
    # Align HTF indicators to 12h timeframe
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_pivot_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or
            np.isnan(camarilla_L3_aligned[i]) or np.isnan(vol_ma_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume (aligned)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        
        # Volume filter: current 1d volume > 1.5 * 20-period mean (volume spike)
        volume_confirmation = vol_1d_aligned[i] > 1.5 * vol_ma_aligned[i]
        
        # Regime filter: chop < 61.8 indicates trending market (avoid choppy ranging)
        regime_filter = chop_aligned[i] < 61.8
        
        # Entry conditions: price breaks Camarilla H3/L3 with volume confirmation and trend regime
        long_entry = (close[i] > camarilla_R3_aligned[i] and volume_confirmation and regime_filter)
        short_entry = (close[i] < camarilla_L3_aligned[i] and volume_confirmation and regime_filter)
        
        # Exit conditions: price returns to Camarilla pivot point (mean reversion to equilibrium)
        long_exit = close[i] < camarilla_pivot_aligned[i]
        short_exit = close[i] > camarilla_pivot_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0