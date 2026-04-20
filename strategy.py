#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for trend and volatility filters
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    close_daily = df_daily['close'].values
    ema34_daily = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Daily ATR (14) for volatility filter
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily_arr = df_daily['close'].values
    
    tr1 = np.abs(high_daily - low_daily)
    tr2 = np.abs(high_daily - np.roll(close_daily_arr, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily_arr, 1))
    tr1[0] = high_daily[0] - low_daily[0]
    tr2[0] = np.abs(high_daily[0] - close_daily_arr[0])
    tr3[0] = np.abs(low_daily[0] - close_daily_arr[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_daily_aligned = align_htf_to_ltf(prices, df_daily, atr_daily)
    
    # 4h data for entry trigger (price action)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA8 for entry timing
    ema8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema34_daily_aligned[i]) or np.isnan(atr_daily_aligned[i]) or 
            np.isnan(ema8[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema34_daily = ema34_daily_aligned[i]
        atr_daily = atr_daily_aligned[i]
        ema8_val = ema8[i]
        
        # Trend filter: align with daily EMA34
        bullish_trend = price > ema34_daily
        bearish_trend = price < ema34_daily
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_daily > 0.001 * price  # at least 0.1% of price
        
        if position == 0:
            # Long: price above daily EMA34 and above 4h EMA8 with adequate volatility
            if bullish_trend and price > ema8_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below daily EMA34 and below 4h EMA8 with adequate volatility
            elif bearish_trend and price < ema8_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below daily EMA34 OR below 4h EMA8
            if not bullish_trend or price < ema8_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above daily EMA34 OR above 4h EMA8
            if not bearish_trend or price > ema8_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_EMA34_EMA8_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0