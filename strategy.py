#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Keltner_Channel_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate daily Keltner Channel (20, 2)
    atr_period = 20
    ma_period = 20
    mult = 2.0
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR
    atr = np.zeros(n)
    atr[:atr_period] = np.nan
    for i in range(atr_period, n):
        if np.isnan(atr[i-1]):
            atr[i] = np.nanmean(tr[i-atr_period+1:i+1])
        else:
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # EMA for middle line
    ema = np.zeros(n)
    ema[:ma_period] = np.nan
    for i in range(ma_period, n):
        if np.isnan(ema[i-1]):
            ema[i] = np.nanmean(close[i-ma_period+1:i+1])
        else:
            ema[i] = (close[i] - ema[i-1]) * (2 / (ma_period + 1)) + ema[i-1]
    
    # Keltner Bands
    upper = ema + (atr * mult)
    lower = ema - (atr * mult)
    
    # Weekly trend: 20-period EMA
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (vol_ma * 1.5)
    
    # Align weekly trend to daily
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(ma_period, atr_period, 20)  # Need enough data
    
    for i in range(start_idx, n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: close above upper Keltner band with volume and above weekly trend
            if close[i] > upper[i] and close[i] > ema20_1w_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: close below lower Keltner band with volume and below weekly trend
            elif close[i] < lower[i] and close[i] < ema20_1w_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below middle line (EMA)
            if close[i] < ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above middle line (EMA)
            if close[i] > ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals