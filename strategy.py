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
    
    # Get weekly data for trend filter (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly EMA(21) for trend filter
    ema21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Daily ATR(14) for volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    tr1_d = df_1d['high'].values - df_1d['low'].values
    tr2_d = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3_d = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    tr_d[0] = tr1_d[0]
    atr_1d_raw = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_raw)
    
    # Daily ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily volume average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup
    start_idx = max(21, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema21_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_daily[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        weekly_trend = ema21_1w_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        atr_daily_val = atr_daily[i]
        vol_ma_val = vol_ma[i]
        
        # Volatility filter: daily ATR > 0.7 * weekly ATR (higher volatility regime)
        vol_filter = atr_daily_val > (atr_1d_val * 0.7)
        
        # Volume filter: current volume > 1.2 * 20-day average
        vol_confirm = volume[i] > (vol_ma_val * 1.2)
        
        if position == 0:
            # Long: price above weekly EMA with volatility and volume confirmation
            if close[i] > weekly_trend and vol_filter and vol_confirm:
                signals[i] = size
                position = 1
            # Short: price below weekly EMA with volatility and volume confirmation
            elif close[i] < weekly_trend and vol_filter and vol_confirm:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below weekly EMA
            if close[i] < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above weekly EMA
            if close[i] > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyEMA21_Trend_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0