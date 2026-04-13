#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h strategy using 12h Camarilla pivot levels + 1d volume spike + 4h ADX trend filter
    # Works in bull/bear: Camarilla provides mean-reversion levels, volume confirms breakout strength,
    # ADX > 25 filters choppy markets. Discrete sizing (0.25) minimizes fee drag.
    # Target: 20-40 trades/year to stay within 4h optimal range.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 12h Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla: H4 = Close + 1.1*(High-Low)/2, L4 = Close - 1.1*(High-Low)/2
    # We use the previous completed 12h bar for pivot calculation
    camarilla_h4 = np.full(len(close_12h), np.nan)
    camarilla_l4 = np.full(len(close_12h), np.nan)
    
    for i in range(1, len(close_12h)):
        # Use previous 12h bar's OHLC
        prev_high = high_12h[i-1]
        prev_low = low_12h[i-1]
        prev_close = close_12h[i-1]
        camarilla_h4[i] = prev_close + 1.1 * (prev_high - prev_low) / 2
        camarilla_l4[i] = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Calculate 1d volume spike (current volume > 2.0 x 20-period average)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = np.zeros_like(volume_1d, dtype=bool)
    volume_spike[20:] = volume_1d[20:] > 2.0 * vol_avg_20_1d[20:]
    
    # Calculate 4h ADX (14-period) for trend filter
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
        if len(high) >= period+1:
            atr[period] = np.mean(tr[1:period+1])
            for i in range(period+1, len(high)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        plus_dm_smoothed = np.zeros_like(high)
        minus_dm_smoothed = np.zeros_like(high)
        if len(high) >= period+1:
            plus_dm_smoothed[period] = np.mean(plus_dm[1:period+1])
            minus_dm_smoothed[period] = np.mean(minus_dm[1:period+1])
            
            for i in range(period+1, len(high)):
                plus_dm_smoothed[i] = (plus_dm_smoothed[i-1] * (period-1) + plus_dm[i]) / period
                minus_dm_smoothed[i] = (minus_dm_smoothed[i-1] * (period-1) + minus_dm[i]) / period
                plus_di[i] = 100 * plus_dm_smoothed[i] / atr[i] if atr[i] != 0 else 0
                minus_di[i] = 100 * minus_dm_smoothed[i] / atr[i] if atr[i] != 0 else 0
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros_like(high)
        if len(high) >= 2*period+1:
            adx[2*period] = np.mean(dx[period+1:2*period+1])
            for i in range(2*period+1, len(high)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_4h = calculate_adx(high, low, close, 14)
    
    # Align all HTF indicators to 4h primary timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    adx_4h_aligned = align_htf_to_ltf(prices, df_12h, adx_4h)  # Using 12h as reference for alignment
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(adx_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 4h ADX > 25 indicates trending market (we fade in chop, follow in trend)
        trending = adx_4h_aligned[i] > 25
        
        # Entry conditions: Camarilla level break + volume spike + trend filter
        # In trending markets: breakout of H4/L4 with volume
        # In ranging markets (ADX <= 25): mean reversion at H4/L4 levels
        enter_long = False
        enter_short = False
        
        if trending:
            # Trending market: breakout strategy
            enter_long = (close[i] > camarilla_h4_aligned[i]) and volume_spike_aligned[i]
            enter_short = (close[i] < camarilla_l4_aligned[i]) and volume_spike_aligned[i]
        else:
            # Ranging market: mean reversion at extremes
            enter_long = (close[i] < camarilla_l4_aligned[i]) and volume_spike_aligned[i]
            enter_short = (close[i] > camarilla_h4_aligned[i]) and volume_spike_aligned[i]
        
        # Stoploss: 1.5x ATR based on 4h true range
        atr_4h = np.zeros_like(high)
        tr_4h = np.zeros_like(high)
        for j in range(1, len(high)):
            tr_4h[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
        if len(high) >= 14:
            atr_4h[13] = np.mean(tr_4h[1:14])
            for j in range(14, len(high)):
                atr_4h[j] = (atr_4h[j-1] * 13 + tr_4h[j]) / 14
        
        # Align ATR to 4h timeframe (it's already 4h)
        atr_4h_aligned = atr_4h
        stop_distance = 1.5 * atr_4h_aligned[i] if not np.isnan(atr_4h_aligned[i]) else (camarilla_h4_aligned[i] - camarilla_l4_aligned[i]) * 0.1
        
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

name = "4h_12h_1d_camarilla_volume_adx_v1"
timeframe = "4h"
leverage = 1.0