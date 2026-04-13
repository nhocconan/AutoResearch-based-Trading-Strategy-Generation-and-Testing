# 12h_1d_wr_breakout_v1
# WR(10) breakout with volume filter and ADX regime filter
# WR measures overbought/oversold conditions
# Long when WR crosses above -50 from oversold in trending market
# Short when WR crosses below -50 from overbought in trending market
# Uses 1d timeframe for WR calculation, aligned to 12h chart
# Volume confirmation reduces false breakouts
# ADX filter ensures we only trade in trending regimes
# Target: 15-25 trades/year, balanced long/short

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
    
    # Get daily data for indicator calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period)
    def calculate_williams_r(high, low, close, period=14):
        highest_high = np.full_like(high, np.nan)
        lowest_low = np.full_like(low, np.nan)
        
        for i in range(len(high)):
            if i >= period - 1:
                start_idx = i - period + 1
                highest_high[i] = np.max(high[start_idx:i+1])
                lowest_low[i] = np.min(low[start_idx:i+1])
        
        wr = np.full_like(close, np.nan)
        for i in range(len(close)):
            if not np.isnan(highest_high[i]) and not np.isnan(lowest_low[i]):
                if highest_high[i] != lowest_low[i]:
                    wr[i] = -100 * (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])
                else:
                    wr[i] = -50  # Avoid division by zero
        return wr
    
    # Calculate WR
    wr_1d = calculate_williams_r(high_1d, low_1d, close_1d, 14)
    
    # Calculate ADX for trend strength (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr = np.full_like(high, np.nan)
        for i in range(len(high)):
            if i == 0:
                tr[i] = high[i] - low[i]
            else:
                tr[i] = max(high[i] - low[i], 
                           abs(high[i] - close[i-1]), 
                           abs(low[i] - close[i-1]))
        
        # Directional Movement
        plus_dm = np.full_like(high, np.nan)
        minus_dm = np.full_like(high, np.nan)
        for i in range(1, len(high)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            else:
                plus_dm[i] = 0
                
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
            else:
                minus_dm[i] = 0
        
        # Smoothed values
        atr = np.full_like(tr, np.nan)
        plus_dm_smooth = np.full_like(plus_dm, np.nan)
        minus_dm_smooth = np.full_like(minus_dm, np.nan)
        
        # Initial values
        if len(tr) >= period:
            atr[period-1] = np.nanmean(tr[:period])
            plus_dm_smooth[period-1] = np.nanmean(plus_dm[:period])
            minus_dm_smooth[period-1] = np.nanmean(minus_dm[:period])
            
            # Wilder's smoothing
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
                minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Directional Indicators
        plus_di = np.full_like(close, np.nan)
        minus_di = np.full_like(close, np.nan)
        dx = np.full_like(close, np.nan)
        
        for i in range(len(atr)):
            if not np.isnan(atr[i]) and atr[i] != 0:
                plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
                minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # ADX (smoothed DX)
        adx = np.full_like(close, np.nan)
        if len(dx) >= period:
            valid_dx = dx[~np.isnan(dx)]
            if len(valid_dx) >= period:
                adx[period-1] = np.nanmean(valid_dx[:period])
                for i in range(period, len(dx)):
                    if not np.isnan(dx[i]):
                        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    # Calculate ADX
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align indicators to 12h timeframe
    wr_1d_aligned = align_htf_to_ltf(prices, df_1d, wr_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume average (20-period)
    volume_ma = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            volume_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(wr_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        volume_confirm = volume[i] > volume_ma[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_1d_aligned[i] > 25
        
        # WR signals: crossing -50 level
        wr_cross_up = (wr_1d_aligned[i] > -50) and (i > 0 and wr_1d_aligned[i-1] <= -50)
        wr_cross_down = (wr_1d_aligned[i] < -50) and (i > 0 and wr_1d_aligned[i-1] >= -50)
        
        # Entry conditions
        long_entry = wr_cross_up and volume_confirm and trending
        short_entry = wr_cross_down and volume_confirm and trending
        
        # Exit conditions: opposite WR cross
        exit_long = position == 1 and wr_cross_down
        exit_short = position == -1 and wr_cross_up
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_wr_breakout_v1"
timeframe = "12h"
leverage = 1.0