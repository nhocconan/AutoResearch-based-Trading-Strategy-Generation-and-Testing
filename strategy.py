#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend and regime filters
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(10) for short-term trend
    close_1w = df_1w['close'].values
    ema_10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Weekly EMA(30) for long-term trend
    ema_30_1w = pd.Series(close_1w).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_30_1w)
    
    # Weekly ATR(14) for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Daily price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily volume filter (current / 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_10_1w_aligned[i]) or np.isnan(ema_30_1w_aligned[i]) or 
            np.isnan(atr_14_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_trend_short = ema_10_1w_aligned[i]
        ema_trend_long = ema_30_1w_aligned[i]
        atr = atr_14_1w_aligned[i]
        vol_ratio_daily = vol_ratio[i]
        
        # Multi-timeframe trend alignment: price above both EMAs for uptrend
        trend_up = (price > ema_trend_short) and (ema_trend_short > ema_trend_long)
        # Price below both EMAs for downtrend
        trend_down = (price < ema_trend_short) and (ema_trend_short < ema_trend_long)
        
        # Volatility filter: avoid low volatility (chop) and extreme volatility
        atr_ma_20 = pd.Series(atr_14_1w_aligned).rolling(window=20, min_periods=20).mean().values[i]
        vol_filter = (atr > 0.5 * atr_ma_20) and (atr < 3.0 * atr_ma_20)
        
        # Volume filter: require above-average volume
        vol_filter = vol_filter and (vol_ratio_daily > 1.3)
        
        if position == 0:
            # Enter long in strong uptrend with volume
            if trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short in strong downtrend with volume
            elif trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend breakdown or volatility spike
            if (not trend_up) or (atr > 3.5 * atr_ma_20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend breakdown or volatility spike
            if (not trend_down) or (atr > 3.5 * atr_ma_20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_EMA10_30_Trend_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0