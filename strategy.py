#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with volume confirmation and 1d ADX regime filter.
# Long when price breaks above Camarilla R3 with volume > 1.8x 20-bar average and ADX > 25 (trending).
# Short when price breaks below Camarilla S3 with volume confirmation and ADX > 25.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Session filter: 08-20 UTC. Target: 20-40 trades/year to minimize fee drag and avoid overtrading.

name = "4h_Camarilla_R3S3_Breakout_Volume_ADX_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for 08-20 UTC filter
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels (R3, S3) from previous day
    def calculate_camarilla(h, l, c):
        """Calculate Camarilla R3 and S3 levels"""
        range_ = h - l
        R3 = c + range_ * 1.1 / 4
        S3 = c - range_ * 1.1 / 4
        return R3, S3
    
    # Calculate ADX(14) for regime filter
    def calculate_adx(high, low, close, period=14):
        """Calculate ADX using Wilder's smoothing"""
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], tr])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed values
        atr_smooth = np.zeros_like(tr)
        dm_plus_smooth = np.zeros_like(dm_plus)
        dm_minus_smooth = np.zeros_like(dm_minus)
        
        # Initial values (simple average)
        atr_smooth[period-1] = np.mean(tr[:period])
        dm_plus_smooth[period-1] = np.mean(dm_plus[:period])
        dm_minus_smooth[period-1] = np.mean(dm_minus[:period])
        
        # Wilder's smoothing
        for i in range(period, len(tr)):
            atr_smooth[i] = (atr_smooth[i-1] * (period-1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / (atr_smooth + 1e-10)
        di_minus = 100 * dm_minus_smooth / (atr_smooth + 1e-10)
        
        # DX and ADX
        dx = np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10) * 100
        adx = np.zeros_like(dx)
        adx[2*period-1] = np.mean(dx[period:2*period])
        
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
            
        return adx
    
    # Calculate ADX
    adx = calculate_adx(high, low, close)
    
    # Calculate Camarilla levels from previous 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R3, S3 for each 1d bar
    camarilla_R3 = np.zeros(len(high_1d))
    camarilla_S3 = np.zeros(len(high_1d))
    
    for i in range(len(high_1d)):
        R3, S3 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        camarilla_R3[i] = R3
        camarilla_S3[i] = S3
    
    # Align Camarilla levels to 4h
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Calculate 1d ADX for regime filter
    adx_1d = calculate_adx(high_1d, low_1d, close_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        if (np.isnan(atr[i]) or np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.8x 20-bar average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        if vol_ma <= 0:
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma * 1.8)
        
        # Camarilla breakout conditions
        breakout_up = curr_high > camarilla_R3_aligned[i]  # break above R3
        breakout_down = curr_low < camarilla_S3_aligned[i]  # break below S3
        
        # ADX regime filter: only trade when trending (ADX > 25)
        adx_filter = adx_1d_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: Camarilla breakout up AND volume confirmation AND ADX trending
            if (breakout_up and 
                volume_confirm and 
                adx_filter):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Camarilla breakout down AND volume confirmation AND ADX trending
            elif (breakout_down and 
                  volume_confirm and 
                  adx_filter):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Camarilla range OR ADX weakens (ranging)
            elif (curr_low <= camarilla_R3_aligned[i] and curr_low >= camarilla_S3_aligned[i]) or \
                 adx_1d_aligned[i] <= 25:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Camarilla range OR ADX weakens (ranging)
            elif (curr_high <= camarilla_R3_aligned[i] and curr_high >= camarilla_S3_aligned[i]) or \
                 adx_1d_aligned[i] <= 25:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals