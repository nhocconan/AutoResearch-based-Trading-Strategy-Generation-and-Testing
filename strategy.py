#!/usr/bin/env python3
"""
6h_WeeklyPivot_Rebound_v1
Hypothesis: In BTC/ETH, price often rebounds from weekly pivot support/resistance levels during ranging markets.
Go long when price touches weekly S1/S2 and shows bullish rejection (close > open) with volume confirmation.
Go short when price touches weekly R1/R2 and shows bearish rejection (close < open) with volume confirmation.
Use 1d trend filter (price vs EMA50) to avoid counter-trend trades. Designed for low-frequency, high-conviction trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using prior week's OHLC)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Weekly pivot: P = (H + L + C) / 3
    pivot_w = (high_w + low_w + close_w) / 3.0
    # Support/resistance levels
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_1w, r2_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_1w, s2_w)
    
    # Daily trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_w_aligned[i]) or np.isnan(r1_w_aligned[i]) or 
            np.isnan(s1_w_aligned[i]) or np.isnan(r2_w_aligned[i]) or
            np.isnan(s2_w_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of daily trend
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Price action rejection signals
        bullish_rejection = close[i] > open_price[i]  # bullish candle
        bearish_rejection = close[i] < open_price[i]  # bearish candle
        
        # Proximity to weekly support/resistance (within 0.5% of level)
        proximity_threshold = 0.005
        near_s1 = abs(low[i] - s1_w_aligned[i]) / s1_w_aligned[i] < proximity_threshold
        near_s2 = abs(low[i] - s2_w_aligned[i]) / s2_w_aligned[i] < proximity_threshold
        near_r1 = abs(high[i] - r1_w_aligned[i]) / r1_w_aligned[i] < proximity_threshold
        near_r2 = abs(high[i] - r2_w_aligned[i]) / r2_w_aligned[i] < proximity_threshold
        
        if position == 0:
            # Long: price near weekly support, bullish rejection, uptrend, volume
            if ((near_s1 or near_s2) and bullish_rejection and uptrend and vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price near weekly resistance, bearish rejection, downtrend, volume
            elif ((near_r1 or near_r2) and bearish_rejection and downtrend and vol_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change, or price reaches weekly pivot/resistance
            if not uptrend or close[i] > pivot_w_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change, or price reaches weekly pivot/support
            if not downtrend or close[i] < pivot_w_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Rebound_v1"
timeframe = "6h"
leverage = 1.0