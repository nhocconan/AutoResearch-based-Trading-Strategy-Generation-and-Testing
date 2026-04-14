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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d RSI(14) for momentum filter
    close_1d = pd.Series(df_1d['close'])
    delta_1d = close_1d.diff()
    gain_1d = delta_1d.where(delta_1d > 0, 0)
    loss_1d = -delta_1d.where(delta_1d < 0, 0)
    avg_gain_1d = gain_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_1d = loss_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_1d = avg_gain_1d / avg_loss_1d
    rsi_1d = (100 - (100 / (1 + rs_1d))).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 4h ATR(14) for volatility filter and position sizing
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 24-period average volume for confirmation (4h * 6 = 24h ~ 1 day)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # ATR-based volatility filter: avoid extremely low volatility periods
        atr_ratio = atr[i] / price if price > 0 else 0
        vol_filter = atr_ratio > 0.003  # Minimum 0.3% ATR relative to price
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = vol > (vol_avg[i] * 1.5) if not np.isnan(vol_avg[i]) else False
        
        # Trend filter: price > 1d EMA50 for long, price < 1d EMA50 for short
        trend_filter_long = price > ema_50_1d_aligned[i]
        trend_filter_short = price < ema_50_1d_aligned[i]
        
        # Momentum filter: RSI between 40 and 60 to avoid extremes and chop
        rsi_filter = 40 <= rsi_1d_aligned[i] <= 60
        
        if position == 0:
            # Long setup: price above 1d EMA50 + volume confirmation + volatility filter + RSI filter
            if (trend_filter_long and vol_confirm and vol_filter and rsi_filter):
                position = 1
                signals[i] = position_size
            # Short setup: price below 1d EMA50 + volume confirmation + volatility filter + RSI filter
            elif (trend_filter_short and vol_confirm and vol_filter and rsi_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 1d EMA50 OR RSI > 70 (overbought)
            if price < ema_50_1d_aligned[i] or rsi_1d_aligned[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 1d EMA50 OR RSI < 30 (oversold)
            if price > ema_50_1d_aligned[i] or rsi_1d_aligned[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1dEMA50_RSI_Volume_Filter"
timeframe = "4h"
leverage = 1.0