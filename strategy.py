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
    
    # Get daily data for price levels and volatility
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate daily 10-period ATR for volatility measurement
    def calculate_atr(high, low, close, period=10):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        atr = np.full_like(tr, np.nan)
        if len(tr) >= period:
            atr[period] = np.nanmean(tr[1:period+1])
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 10)
    
    # Calculate weekly EMA(34) for trend filter
    if len(close_1w) >= 34:
        ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False).mean().values
    else:
        ema_1w = np.full_like(close_1w, np.nan)
    
    # Calculate 6-period range (high-low) for volatility breakout detection
    range_6 = np.full(n, np.nan)
    for i in range(6, n):
        range_6[i] = np.max(high[i-6:i]) - np.min(low[i-6:i])
    
    # Calculate 6-period ATR equivalent from daily ATR
    # Convert daily ATR to 6h equivalent (assuming 4x 6h bars per day)
    atr_6h_equiv = atr_1d / 2  # Conservative estimate
    
    # Align all data to 6h timeframe
    atr_6h_aligned = align_htf_to_ltf(prices, df_1d, atr_6h_equiv)
    ema_1w_6h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.5x 24-period average (4 days of 6h bars)
    vol_ma = np.full(n, np.nan)
    vol_period = 24
    if n >= vol_period:
        for i in range(vol_period, n):
            vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24, 6) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_6h_aligned[i]) or np.isnan(ema_1w_6h[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(range_6[i])):
            signals[i] = 0.0
            continue
        
        # Volatility breakout: current 6h range > 1.5x average volatility
        vol_breakout = range_6[i] > 1.5 * atr_6h_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below weekly EMA
        above_weekly_ema = close[i] > ema_1w_6h[i]
        below_weekly_ema = close[i] < ema_1w_6h[i]
        
        if position == 0:
            # Long: volatility breakout to upside with volume in bullish trend
            if (close[i] > np.max(high[i-6:i]) and vol_breakout and vol_confirm and above_weekly_ema):
                signals[i] = 0.25
                position = 1
            # Short: volatility breakout to downside with volume in bearish trend
            elif (close[i] < np.min(low[i-6:i]) and vol_breakout and vol_confirm and below_weekly_ema):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below 6h low OR volatility breaks down
            if (close[i] < np.min(low[i-6:i]) or not vol_breakout):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above 6h high OR volatility breaks down
            if (close[i] > np.max(high[i-6:i]) or not vol_breakout):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_VolatilityBreakout_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0