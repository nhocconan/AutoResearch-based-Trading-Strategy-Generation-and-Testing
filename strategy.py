#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d RSI (14-period) - Wilder's smoothing ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
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
    
    # === Align indicators to daily timeframe ===
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # === Weekly Volume Confirmation ===
    df_1w = get_htf_data(prices, '1w')
    volume_1w = df_1w['volume'].values
    
    # Calculate 4-week average volume on weekly timeframe
    vol_ma_4 = np.full_like(volume_1w, np.nan)
    for i in range(len(volume_1w)):
        if i >= 3:
            vol_ma_4[i] = np.mean(volume_1w[i-3:i+1])
        elif i > 0:
            vol_ma_4[i] = np.mean(volume_1w[max(0, i-1):i+1])
        else:
            vol_ma_4[i] = volume_1w[0]
    
    # Volume confirmation: current weekly volume > 1.5x 4-week average
    vol_confirm_1w = volume_1w > vol_ma_4 * 1.5
    vol_confirm_aligned = align_htf_to_ltf(prices, df_1w, vol_confirm_1w)
    
    # === Session filter (08-20 UTC) ===
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_width_percentile_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or 
            np.isnan(vol_confirm_aligned[i])):
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
                vol_confirm_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: RSI > 70 (overbought) + BB squeeze + price near upper BB + volume confirmation
            elif (rsi_1d_aligned[i] > 70 and 
                  is_squeeze and 
                  close[i] >= upper_bb_aligned[i] * 0.98 and  # within 2% of upper BB
                  vol_confirm_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: RSI crosses above 50 OR price reaches middle BB
            if rsi_1d_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses below 50 OR price reaches middle BB
            if rsi_1d_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_BB_Squeeze_RSI_MeanReversion_WeeklyVolumeFilter_v1"
timeframe = "1d"
leverage = 1.0