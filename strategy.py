#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA(21) for trend filter
    ema_21_12h = pd.Series(df_12h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Calculate 12h ATR(14) for volatility filter
    high_12h = pd.Series(df_12h['high'])
    low_12h = pd.Series(df_12h['low'])
    close_12h = pd.Series(df_12h['close'])
    tr1_12h = high_12h - low_12h
    tr2_12h = abs(high_12h - close_12h.shift(1))
    tr3_12h = abs(low_12h - close_12h.shift(1))
    tr_12h = pd.concat([tr1_12h, tr2_12h, tr3_12h], axis=1).max(axis=1)
    atr_12h = tr_12h.rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate 12h RSI(14) for momentum filter
    delta_12h = close_12h.diff()
    gain_12h = delta_12h.clip(lower=0)
    loss_12h = -delta_12h.clip(upper=0)
    avg_gain_12h = gain_12h.rolling(window=14, min_periods=14).mean()
    avg_loss_12h = loss_12h.rolling(window=14, min_periods=14).mean()
    rs_12h = avg_gain_12h / avg_loss_12h
    rsi_12h = (100 - (100 / (1 + rs_12h))).values
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_21_12h_aligned[i]) or 
            np.isnan(atr_12h_aligned[i]) or
            np.isnan(rsi_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volatility filter: avoid extremely low volatility periods
        atr_ratio = atr_12h_aligned[i] / price if price > 0 else 0
        vol_filter = atr_ratio > 0.008  # Minimum 0.8% ATR relative to price
        
        # Trend filter: price > 12h EMA21 for long, price < 12h EMA21 for short
        trend_filter_long = price > ema_21_12h_aligned[i]
        trend_filter_short = price < ema_21_12h_aligned[i]
        
        # Momentum filter: RSI between 35 and 65 to avoid extremes
        rsi_filter = (rsi_12h_aligned[i] > 35) & (rsi_12h_aligned[i] < 65)
        
        # Volume filter: above average volume
        vol_ma = np.mean(volume[max(0, i-20):i+1]) if i >= 20 else volume[i]
        vol_filter_enhanced = vol_filter and (volume[i] > vol_ma * 0.7)
        
        if position == 0:
            # Long setup: price above 12h EMA21 + volatility filter + momentum filter
            if (trend_filter_long and vol_filter_enhanced and rsi_filter):
                position = 1
                signals[i] = position_size
            # Short setup: price below 12h EMA21 + volatility filter + momentum filter
            elif (trend_filter_short and vol_filter_enhanced and rsi_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 12h EMA21
            if price < ema_21_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 12h EMA21
            if price > ema_21_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_EMA21_ATR_RSI_Filter_v1"
timeframe = "12h"
leverage = 1.0