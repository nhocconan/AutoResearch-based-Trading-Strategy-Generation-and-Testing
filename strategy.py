#!/usr/bin/env python3
"""
Experiment #1635: 6h Camarilla Pivot Reversal + 1w Trend Filter + Volume Spike
HYPOTHESIS: Camarilla pivot levels (R3/S3, R4/S4) from 1d timeframe act as institutional support/resistance. 
In ranging markets (CHOP > 50), price tends to reverse from R3/S3 levels. In trending markets (CHOP <= 50), 
breaks of R4/S4 with 1w trend alignment and volume confirmation (>2x average) signal continuation. 
This combines mean reversion in ranges with trend-following breakouts, adapting to market regime. 
Target: 75-150 total trades over 4 years (19-37/year) with discrete position sizing to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1635_6h_camarilla_1w_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # EMA(21) for 1w trend
    ema_1w = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    trend_1w = np.where(close_1w > ema_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === HTF: 1d data for Camarilla pivots (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    rang = high_1d - low_1d
    r4_1d = close_1d + (rang * 1.1 / 2)
    r3_1d = close_1d + (rang * 1.1 / 4)
    s3_1d = close_1d - (rang * 1.1 / 4)
    s4_1d = close_1d - (rang * 1.1 / 2)
    
    # Align to 6h timeframe (shifted by 1 for completed bars only)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === 6h Indicators: Choppiness Index for regime detection ===
    def choppiness_index(high_arr, low_arr, close_arr, period=14):
        """Calculate Choppiness Index: higher = ranging, lower = trending"""
        atr_sum = np.zeros(n)
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(high_arr[i] - low_arr[i], 
                       abs(high_arr[i] - close_arr[i-1]), 
                       abs(low_arr[i] - close_arr[i-1]))
        tr[0] = high_arr[0] - low_arr[0]
        # Sum of TRUE range over period
        atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
        # Highest high and lowest low over period
        hh = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        ll = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        # Choppiness formula
        chop = np.zeros(n)
        mask = (hh - ll) > 0
        chop[mask] = 100 * np.log10(atr_sum[mask] / (hh[mask] - ll[mask])) / np.log10(period)
        chop[~mask] = 50  # neutral when no range
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    # Regime: CHOP > 50 = ranging (mean revert), CHOP <= 50 = trending (breakout)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 20  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(trend_1w_aligned[i]) or np.isnan(chop[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Reverse signal or stoploss via signal=0 ---
        if in_position:
            # Exit conditions: reverse signal or adverse move
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if: price reaches R3 (take profit) OR breaks below S3 (stop)
                if price >= r3_1d_aligned[i] or price <= s3_1d_aligned[i]:
                    exit_signal = True
            else:  # Short position
                # Exit if: price reaches S3 (take profit) OR breaks above R3 (stop)
                if price <= s3_1d_aligned[i] or price >= r3_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                signals[i] = 0.0
                continue
            else:
                signals[i] = position_side * SIZE
                continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require significant spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            if chop[i] > 50:  # Ranging market: mean reversion at R3/S3
                # Long: price rejects S3 and moves back above it
                if price > s3_1d_aligned[i] and low[i] <= s3_1d_aligned[i]:
                    in_position = True
                    position_side = 1
                    signals[i] = SIZE
                # Short: price rejects R3 and moves back below it
                elif price < r3_1d_aligned[i] and high[i] >= r3_1d_aligned[i]:
                    in_position = True
                    position_side = -1
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:  # Trending market: breakout continuation at R4/S4
                # Require 1w trend alignment
                if trend_1w_aligned[i] > 0:  # Uptrend
                    if price > r4_1d_aligned[i]:  # Break above R4
                        in_position = True
                        position_side = 1
                        signals[i] = SIZE
                    else:
                        signals[i] = 0.0
                else:  # Downtrend
                    if price < s4_1d_aligned[i]:  # Break below S4
                        in_position = True
                        position_side = -1
                        signals[i] = -SIZE
                    else:
                        signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals