#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h strategy using 1d Camarilla pivot levels with volume confirmation and 1w ADX trend filter
    # Works in both bull and bear: Camarilla levels provide structured support/resistance,
    # 1w ADX > 20 filters choppy markets, volume confirmation ensures breakout validity.
    # Discrete sizing (0.25) minimizes fee drag. Target: 12-30 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for Camarilla pivots (primary HTF for levels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    camarilla_h4 = np.zeros_like(close_1d)
    camarilla_l4 = np.zeros_like(close_1d)
    camarilla_h3 = np.zeros_like(close_1d)
    camarilla_l3 = np.zeros_like(close_1d)
    camarilla_h2 = np.zeros_like(close_1d)
    camarilla_l2 = np.zeros_like(close_1d)
    camarilla_h1 = np.zeros_like(close_1d)
    camarilla_l1 = np.zeros_like(close_1d)
    pivot = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        # Previous day's OHLC
        phigh = high_1d[i-1]
        plow = low_1d[i-1]
        pclose = close_1d[i-1]
        range_val = phigh - plow
        
        # Camarilla equations
        pivot[i] = (phigh + plow + pclose) / 3
        camarilla_h1[i] = pclose + range_val * 1.1 / 12
        camarilla_l1[i] = pclose - range_val * 1.1 / 12
        camarilla_h2[i] = pclose + range_val * 1.1 / 6
        camarilla_l2[i] = pclose - range_val * 1.1 / 6
        camarilla_h3[i] = pclose + range_val * 1.1 / 4
        camarilla_l3[i] = pclose - range_val * 1.1 / 4
        camarilla_h4[i] = pclose + range_val * 1.1 / 2
        camarilla_l4[i] = pclose - range_val * 1.1 / 2
    
    # Calculate 1w ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        plus_dm_smoothed = np.zeros_like(high)
        minus_dm_smoothed = np.zeros_like(high)
        plus_dm_smoothed[period] = np.mean(plus_dm[1:period+1])
        minus_dm_smoothed[period] = np.mean(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            plus_dm_smoothed[i] = (plus_dm_smoothed[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smoothed[i] = (minus_dm_smoothed[i-1] * (period-1) + minus_dm[i]) / period
            plus_di[i] = 100 * plus_dm_smoothed[i] / atr[i]
            minus_di[i] = 100 * minus_dm_smoothed[i] / atr[i]
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) if (plus_di[i] + minus_di[i]) != 0 else 0
        
        adx = np.zeros_like(high)
        adx[2*period] = np.mean(dx[period+1:2*period+1])
        for i in range(2*period+1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Get 1d volume for confirmation (20-period average)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 12h primary timeframe
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h1_aligned[i]) or 
            np.isnan(camarilla_l1_aligned[i]) or
            np.isnan(adx_1w_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        idx_1d = i // (24 * 2)  # 1d bars in 12h timeframe (2 bars per day)
        if idx_1d >= len(volume_1d):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_1d[idx_1d] > 1.3 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: 1w ADX > 20 indicates trending market (not choppy)
        trending = adx_1w_aligned[i] > 20
        
        # Entry conditions: Camarilla H3/L3 breakout + trend + volume
        # H3/L3 are stronger levels than H1/L1, reducing false breakouts
        enter_long = (close[i] > camarilla_h3_aligned[i]) and trending and volume_confirmed
        enter_short = (close[i] < camarilla_l3_aligned[i]) and trending and volume_confirmed
        
        # Stoploss: 1.5x ATR based on Camarilla H3-L3 width
        camarilla_width = camarilla_h3_aligned[i] - camarilla_l3_aligned[i]
        stop_distance = camarilla_width * 0.15  # 15% of level width
        
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - stop_distance
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + stop_distance
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "12h_1d_1w_camarilla_breakout_adx_volume_v1"
timeframe = "12h"
leverage = 1.0