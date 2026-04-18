#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mti_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for weekly trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(34) for trend filter
    if len(close_1w) >= 34:
        ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    else:
        ema_1w = np.full_like(close_1w, np.nan)
    
    # Align weekly EMA to daily timeframe
    ema_1w_daily = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # ATR(14) for volatility filtering
    def calculate_atr(high, low, close, period=14):
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
    
    atr_1d = calculate_atr(high, low, close, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Daily EMA(50) for trend filter
    if len(close_1d) >= 50:
        ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema_50 = np.full_like(close_1d, np.nan)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Price position relative to EMA50 (trend strength)
    price_above_ema50 = close > ema_50_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_1w_daily[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr_1d_aligned[i] > 0.01 * close[i]  # ATR > 1% of price
        
        if position == 0:
            # Long: price above weekly EMA AND daily EMA50 with sufficient volatility
            if close[i] > ema_1w_daily[i] and close[i] > ema_50_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA AND daily EMA50 with sufficient volatility
            elif close[i] < ema_1w_daily[i] and close[i] < ema_50_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below daily EMA50 OR volatility drops too low
            if close[i] < ema_50_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above daily EMA50 OR volatility drops too low
            if close[i] > ema_50_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_EMA50_WeeklyTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0