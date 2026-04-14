#!/usr/bin/env python3
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
    
    # Pre-compute hour for session filter (08-20 UTC)
    hours = prices.index.hour.values  # already datetime64[ms]
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h ATR (14-period) for volatility filter
    high_12h = pd.Series(df_12h['high'])
    low_12h = pd.Series(df_12h['low'])
    close_12h = pd.Series(df_12h['close'])
    tr1_12h = high_12h - low_12h
    tr2_12h = abs(high_12h - close_12h.shift(1))
    tr3_12h = abs(low_12h - close_12h.shift(1))
    tr_12h = pd.concat([tr1_12h, tr2_12h, tr3_12h], axis=1).max(axis=1)
    atr_12h = tr_12h.rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate 12h EMA(50) for trend filter
    ema_50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h RSI for momentum confirmation
    delta_12h = close_12h.diff()
    gain_12h = delta_12h.where(delta_12h > 0, 0)
    loss_12h = -delta_12h.where(delta_12h < 0, 0)
    avg_gain_12h = gain_12h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_12h = loss_12h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_12h = avg_gain_12h / avg_loss_12h
    rsi_12h = (100 - (100 / (1 + rs_12h))).values
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Calculate 60-period rolling average volume for 6h timeframe
    vol_avg_60 = pd.Series(volume).rolling(window=60, min_periods=60).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Session filter: only trade 08-20 UTC
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any critical data is NaN
        if (np.isnan(atr_12h_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(rsi_12h_aligned[i]) or
            np.isnan(vol_avg_60[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # ATR-based volatility filter: avoid extremely low volatility periods
        atr_ratio = atr_12h_aligned[i] / price if price > 0 else 0
        vol_filter = atr_ratio > 0.003  # Minimum 0.3% ATR relative to price
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = vol > (vol_avg_60[i] * 1.5) if not np.isnan(vol_avg_60[i]) else False
        
        # Trend filter: price > 12h EMA50 for long, price < 12h EMA50 for short
        trend_filter_long = price > ema_50_12h_aligned[i]
        trend_filter_short = price < ema_50_12h_aligned[i]
        
        # Momentum filter: RSI between 40 and 60 to avoid extremes
        rsi_filter = 40 <= rsi_12h_aligned[i] <= 60
        
        if position == 0:
            # Long setup: price above 12h EMA50 + volume confirmation + volatility filter + RSI filter
            if (trend_filter_long and vol_confirm and vol_filter and rsi_filter):
                position = 1
                signals[i] = position_size
            # Short setup: price below 12h EMA50 + volume confirmation + volatility filter + RSI filter
            elif (trend_filter_short and vol_confirm and vol_filter and rsi_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 12h EMA50
            if price < ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 12h EMA50
            if price > ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12hEMA50_Volume_RSI_Filter"
timeframe = "6h"
leverage = 1.0