#!/usr/bin/env python3
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
    
    # === 1d RSI (14-period) - Wilder's smoothing ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI using Wilder's smoothing with proper seeding
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    period = 14
    for i in range(len(gain)):
        if i < period:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (i-1) + gain[i]) / i
                avg_loss[i] = (avg_loss[i-1] * (i-1) + loss[i]) / i
        else:
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[avg_loss == 0] = 100
    
    # === 1d Bollinger Bands (20,2) ===
    # Calculate SMA20
    sma_20 = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i >= 19:
            sma_20[i] = np.mean(close_1d[i-19:i+1])
        elif i > 0:
            sma_20[i] = np.mean(close_1d[max(0, i-9):i+1])
        else:
            sma_20[i] = close_1d[0]
    
    # Calculate standard deviation
    std_20 = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i >= 19:
            std_20[i] = np.std(close_1d[i-19:i+1])
        elif i > 0:
            std_20[i] = np.std(close_1d[max(0, i-9):i+1])
        else:
            std_20[i] = 0.0
    
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # === 1d Bollinger Band Width (for squeeze detection) ===
    bb_width = (upper_bb - lower_bb) / sma_20
    
    # === 1d BB Width percentile (20-period) for regime detection ===
    bb_width_percentile = np.full_like(bb_width, np.nan)
    for i in range(len(bb_width)):
        if i >= 19:
            window = bb_width[i-19:i+1]
            rank = np.sum(window <= bb_width[i]) / len(window)
            bb_width_percentile[i] = rank * 100
        elif i > 0:
            window = bb_width[max(0, i-9):i+1]
            rank = np.sum(window <= bb_width[i]) / len(window)
            bb_width_percentile[i] = rank * 100
        else:
            bb_width_percentile[i] = 50.0
    
    # === Align indicators to 1h timeframe ===
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20)
    
    # === 4h Volume confirmation ===
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    
    # Calculate 20-period average volume on 4h timeframe
    vol_ma_20 = np.full_like(volume_4h, np.nan)
    for i in range(len(volume_4h)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume_4h[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume_4h[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume_4h[0]
    
    # Volume confirmation: current 4h volume > 1.3x 20-period average
    vol_confirm = volume_4h > vol_ma_20 * 1.3
    
    # === 1h Session filter (08-20 UTC) ===
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_width_percentile_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or 
            np.isnan(sma_20_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if outside session
        if not session_filter[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Market regime: low volatility squeeze (BB Width < 20th percentile)
        is_squeeze = bb_width_percentile_aligned[i] < 20
        
        # Entry logic: only enter when flat AND volume confirmation
        if position == 0:
            # Long: RSI < 30 (oversold) + BB squeeze + price near lower BB + volume confirmation
            if (rsi_1d_aligned[i] < 30 and 
                is_squeeze and 
                close[i] <= lower_bb_aligned[i] * 1.02 and  # within 2% of lower BB
                vol_confirm[i]):
                signals[i] = 0.20
                position = 1
                continue
            # Short: RSI > 70 (overbought) + BB squeeze + price near upper BB + volume confirmation
            elif (rsi_1d_aligned[i] > 70 and 
                  is_squeeze and 
                  close[i] >= upper_bb_aligned[i] * 0.98 and  # within 2% of upper BB
                  vol_confirm[i]):
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: RSI crosses above 50 OR price reaches middle BB
            if (rsi_1d_aligned[i] > 50 or 
                close[i] >= sma_20_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI crosses below 50 OR price reaches middle BB
            if (rsi_1d_aligned[i] < 50 or 
                close[i] <= sma_20_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_BB_Squeeze_RSI_MeanReversion_VolumeFilter_v2"
timeframe = "1h"
leverage = 1.0