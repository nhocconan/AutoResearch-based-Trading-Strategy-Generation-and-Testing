#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume confirmation and ADX trend filter.
# Camarilla levels provide high-probability reversal/breakout zones.
# Volume confirms conviction; ADX ensures alignment with trend.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 4-period ATR for volatility (used in ADX)
    atr_period = 4
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # Calculate ADX (14-period) for trend strength
    adx_period = 14
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smooth DM and TR
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    if n >= adx_period:
        plus_dm_smooth = np.zeros(n)
        minus_dm_smooth = np.zeros(n)
        atr_smooth = np.zeros(n)
        
        # Initial smoothed values
        plus_dm_smooth[adx_period-1] = np.sum(plus_dm[1:adx_period+1])
        minus_dm_smooth[adx_period-1] = np.sum(minus_dm[1:adx_period+1])
        atr_smooth[adx_period-1] = np.sum(tr[1:adx_period+1])
        
        # Wilder's smoothing
        for i in range(adx_period, n):
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / adx_period) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / adx_period) + minus_dm[i]
            atr_smooth[i] = atr_smooth[i-1] - (atr_smooth[i-1] / adx_period) + tr[i]
        
        # Calculate DI
        for i in range(adx_period-1, n):
            if atr_smooth[i] != 0:
                plus_di[i] = 100 * (plus_dm_smooth[i] / atr_smooth[i])
                minus_di[i] = 100 * (minus_dm_smooth[i] / atr_smooth[i])
        
        # Calculate DX and ADX
        dx = np.zeros(n)
        for i in range(adx_period-1, n):
            if (plus_di[i] + minus_di[i]) != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros(n)
        if n >= 2 * adx_period - 1:
            adx[2*adx_period-2] = np.sum(dx[adx_period-1:2*adx_period-1]) / adx_period
            for i in range(2*adx_period-1, n):
                adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period
    
    # Calculate average volume (20-period) for volume confirmation
    avg_volume = np.zeros(n)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Calculate Camarilla levels from previous day
    camarilla_H4 = np.full(n, np.nan)
    camarilla_L4 = np.full(n, np.nan)
    camarilla_H3 = np.full(n, np.nan)
    camarilla_L3 = np.full(n, np.nan)
    
    for i in range(1, len(df_1d)):
        # Previous day's OHLC
        prev_high = df_1d['high'].iloc[i-1]
        prev_low = df_1d['low'].iloc[i-1]
        prev_close = df_1d['close'].iloc[i-1]
        
        # Camarilla calculations
        range_val = prev_high - prev_low
        camarilla_H4[i] = prev_close + range_val * 1.1 / 2
        camarilla_L4[i] = prev_close - range_val * 1.1 / 2
        camarilla_H3[i] = prev_close + range_val * 1.1 / 4
        camarilla_L3[i] = prev_close - range_val * 1.1 / 4
        
        # Propagate levels to intraday periods (assuming 6 bars per day)
        bars_per_day = 6
        start_idx = i * bars_per_day
        end_idx = start_idx + bars_per_day
        if end_idx <= n:
            camarilla_H4[start_idx:end_idx] = camarilla_H4[i]
            camarilla_L4[start_idx:end_idx] = camarilla_L4[i]
            camarilla_H3[start_idx:end_idx] = camarilla_H3[i]
            camarilla_L3[start_idx:end_idx] = camarilla_L3[i]
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_H4[i]) or np.isnan(camarilla_L4[i]) or 
            np.isnan(camarilla_H3[i]) or np.isnan(camarilla_L3[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(adx[i]) if i < len(adx) else True):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        adx_val = adx[i] if i < len(adx) else 0
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        # ADX filter: trend strength > 20
        adx_filter = adx_val > 20
        
        if position == 0:
            # Long: price breaks above H3 with volume and trend
            if (price > camarilla_H3[i] and 
                volume_confirm and 
                adx_filter):
                position = 1
                signals[i] = position_size
            # Short: price breaks below L3 with volume and trend
            elif (price < camarilla_L3[i] and 
                  volume_confirm and 
                  adx_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below L3 or volume drops significantly
            if (price < camarilla_L3[i] or 
                vol < 0.4 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above H3 or volume drops significantly
            if (price > camarilla_H3[i] or 
                vol < 0.4 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_Breakout_Volume_ADX_v1"
timeframe = "4h"
leverage = 1.0