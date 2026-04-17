#!/usr/bin/env python3
"""
Hypothesis: 1h Volume-Weighted RSI + 4h Trend Filter + Session.
Long when 1h VW-RSI < 30 and 4h close > 4h EMA50 with volume confirmation.
Short when 1h VW-RSI > 70 and 4h close < 4h EMA50 with volume confirmation.
Exit on opposite RSI extreme or volume divergence.
Uses 4h for trend/EMA filter, 1h for entry timing with VW-RSI.
Target: 60-150 total trades over 4 years (15-37/year).
"""

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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h EMA50
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h volume MA20 for confirmation
    vol_ma20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align 4h indicators to 1h
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    vol_ma20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma20_4h)
    
    # Calculate 1h VW-RSI (14-period)
    def calculate_vw_rsi(close, high, low, volume, period=14):
        typical_price = (high + low + close) / 3.0
        # Volume-weighted price change
        vwap_change = typical_price * volume
        
        # Calculate gains and losses
        delta = np.diff(typical_price, prepend=typical_price[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Volume-weighted gains and losses
        vw_gain = gain * volume
        vw_loss = loss * volume
        
        # Wilder's smoothing with volume weighting
        avg_vw_gain = np.zeros_like(vw_gain)
        avg_vw_loss = np.zeros_like(vw_loss)
        
        avg_vw_gain[period] = np.mean(vw_gain[1:period+1])
        avg_vw_loss[period] = np.mean(vw_loss[1:period+1])
        
        for i in range(period+1, len(vw_gain)):
            avg_vw_gain[i] = (avg_vw_gain[i-1] * (period-1) + vw_gain[i]) / period
            avg_vw_loss[i] = (avg_vw_loss[i-1] * (period-1) + vw_loss[i]) / period
        
        # Avoid division by zero
        rs = np.where(avg_vw_loss != 0, avg_vw_gain / avg_vw_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    vw_rsi = calculate_vw_rsi(close, high, low, volume, 14)
    
    # Volume confirmation on 1h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_ma20_4h_aligned[i]) or
            np.isnan(vw_rsi[i]) or
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # 4h trend and volume conditions
        price_4h_close = close_4h[min(i // 16, len(close_4h)-1)] if i >= 16 else close_4h[0]
        ema50_val = ema50_4h_aligned[i]
        vol_4h_current = volume_4h[min(i // 16, len(volume_4h)-1)] if i >= 16 else volume_4h[0]
        vol_ma20_4h_val = vol_ma20_4h_aligned[i]
        
        is_uptrend = price_4h_close > ema50_val and vol_4h_current > vol_ma20_4h_val
        is_downtrend = price_4h_close < ema50_val and vol_4h_current > vol_ma20_4h_val
        
        # 1h VW-RSI and volume conditions
        rsi_val = vw_rsi[i]
        vol_current = volume[i]
        vol_ma20_val = vol_ma20[i]
        is_high_volume = vol_current > vol_ma20_val
        
        if position == 0:
            # Long: Oversold VW-RSI + 4h uptrend + volume confirmation
            if rsi_val < 30 and is_uptrend and is_high_volume:
                signals[i] = 0.20
                position = 1
            # Short: Overbought VW-RSI + 4h downtrend + volume confirmation
            elif rsi_val > 70 and is_downtrend and is_high_volume:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Overbought VW-RSI or 4h trend breaks
            if rsi_val > 70 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Oversold VW-RSI or 4h trend breaks
            if rsi_val < 30 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_VW_RSI_4hEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0