#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Breakout_Volume
Hypothesis: Use 1d Camarilla pivot levels (S1/S2 for longs, R1/R2 for shorts) with volume confirmation and ADX trend filter. Go long when price breaks above S1/S2 with volume > 1.5x 20-period average and ADX > 25 (trending). Go short when price breaks below R1/R2 with same filters. In ranging markets (ADX < 20), fade the extremes: long at S3, short at R3. This structure works in bull markets by following breakouts, and in bear markets by fading overextended moves at daily extremes. Targets 15-25 trades/year by requiring multiple confluence factors.
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
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels
    S1 = close_1d - (range_hl * 1.1 / 6)
    S2 = close_1d - (range_hl * 1.1 / 4)
    S3 = close_1d - (range_hl * 1.1 / 2)
    R1 = close_1d + (range_hl * 1.1 / 6)
    R2 = close_1d + (range_hl * 1.1 / 4)
    R3 = close_1d + (range_hl * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe (wait for day close)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    # ADX trend filter (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        adx = np.zeros_like(high)
        
        if len(high) >= period:
            atr[period-1] = np.mean(tr[:period])
            plus_dm_sum = np.sum(plus_dm[:period])
            minus_dm_sum = np.sum(minus_dm[:period])
            
            for i in range(period, len(high)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_dm_sum = plus_dm_sum - (plus_dm_sum / period) + plus_dm[i]
                minus_dm_sum = minus_dm_sum - (minus_dm_sum / period) + minus_dm[i]
                plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
                minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
                dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100 if (plus_di[i] + minus_di[i]) != 0 else 0
            
            if len(high) >= 2*period-1:
                adx[2*period-2] = np.mean(dx[period-1:2*period-1])
                for i in range(2*period-1, len(high)):
                    adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need ADX and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(S1_aligned[i]) or np.isnan(S2_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(R1_aligned[i]) or np.isnan(R2_aligned[i]) or np.isnan(R3_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Trending market (ADX > 25): breakout strategy
            if adx[i] > 25:
                # Long: price breaks above S1 or S2 with volume
                if ((close[i] > S1_aligned[i] or close[i] > S2_aligned[i]) and vol_confirm[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below R1 or R2 with volume
                elif ((close[i] < R1_aligned[i] or close[i] < R2_aligned[i]) and vol_confirm[i]):
                    signals[i] = -0.25
                    position = -1
            # Ranging market (ADX < 20): fade extremes
            else:
                # Long at S3 (strong support)
                if close[i] <= S3_aligned[i] and vol_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                # Short at R3 (strong resistance)
                elif close[i] >= R3_aligned[i] and vol_confirm[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot or ADX drops (trend weakening)
            if (close[i] <= pivot[np.searchsorted(df_1d.index, pd.Timestamp(prices.iloc[i]['open_time']))] if 
                np.searchsorted(df_1d.index, pd.Timestamp(prices.iloc[i]['open_time'])) < len(df_1d) else 
                pivot[-1]) or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot or ADX drops
            if (close[i] >= pivot[np.searchsorted(df_1d.index, pd.Timestamp(prices.iloc[i]['open_time']))] if 
                np.searchsorted(df_1d.index, pd.Timestamp(prices.iloc[i]['open_time'])) < len(df_1d) else 
                pivot[-1]) or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_Breakout_Volume"
timeframe = "12h"
leverage = 1.0