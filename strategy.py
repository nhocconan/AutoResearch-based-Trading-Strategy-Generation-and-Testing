#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with 12-hour EMA trend filter and volume confirmation.
# Uses daily ATR to filter volatility regime and 12-hour EMA34 for trend direction.
# Enters on 4h Donchian(20) breakouts above/below with volume confirmation.
# Designed for 20-40 trades per year to avoid fee drag in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12-hour data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(34) on 12h data
    ema_34_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 34:
        ema_34_12h[33] = np.mean(close_12h[:34])
        for i in range(34, len(close_12h)):
            ema_34_12h[i] = (close_12h[i] * 2 / (34 + 1)) + (ema_34_12h[i-1] * (33 / (34 + 1)))
    
    # Align 12h EMA34 to 4h timeframe
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Get daily data for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on daily data with proper min_periods
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align daily ATR14 to 4h timeframe
    atr_14d_aligned = align_htf_to_ltf(prices, df_1d, atr_14d)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Calculate Donchian channels (20-period) on 4h data
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    for i in range(20, n):
        upper_channel[i] = np.max(high[i-20:i])
        lower_channel[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 14)  # need EMA34, Donchian, ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(atr_14d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(upper_channel[i]) or np.isnan(lower_channel[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Volatility filter: ATR14d > 0.5% of price (avoid low volatility)
        vol_filter = atr_14d_aligned[i] > 0.005 * close[i]
        
        # Trend filter: price above/below 12h EMA34
        trend_up = close[i] > ema_34_12h_aligned[i]
        trend_down = close[i] < ema_34_12h_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian upper channel with volume and trend
            if (close[i] > upper_channel[i] and 
                vol_confirmed and 
                vol_filter and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower channel with volume and trend
            elif (close[i] < lower_channel[i] and 
                  vol_confirmed and 
                  vol_filter and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below Donchian lower channel or ATR-based stop
            if close[i] < lower_channel[i] or close[i] < (upper_channel[i] - 2.0 * atr_14d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian upper channel or ATR-based stop
            if close[i] > upper_channel[i] or close[i] > (lower_channel[i] + 2.0 * atr_14d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA34_VolumeFilter_ATR"
timeframe = "4h"
leverage = 1.0