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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily ATR (14-period) for volatility and stop loss
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate weekly EMA(20) for higher timeframe trend
    df_1w = get_htf_data(prices, '1w')
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate RSI(14) for momentum
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs)).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        price = close[i]
        vol = volume[i]
        
        # Skip if any critical data is NaN
        if (np.isnan(atr[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().iloc[i]
        vol_confirm = vol > (vol_avg * 1.5) if not np.isnan(vol_avg) else False
        
        # Trend alignment: daily EMA50 > weekly EMA20 for bullish, < for bearish
        daily_above_weekly = ema_50_1d_aligned[i] > ema_20_1w_aligned[i]
        daily_below_weekly = ema_50_1d_aligned[i] < ema_20_1w_aligned[i]
        
        # Momentum filter: RSI between 40 and 60 to avoid extremes
        rsi_filter = 40 <= rsi[i] <= 60
        
        if position == 0:
            # Long setup: daily EMA above weekly EMA + volume confirmation + RSI filter
            if daily_above_weekly and vol_confirm and rsi_filter:
                position = 1
                signals[i] = position_size
            # Short setup: daily EMA below weekly EMA + volume confirmation + RSI filter
            elif daily_below_weekly and vol_confirm and rsi_filter:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: daily EMA crosses below weekly EMA OR stop loss hit
            if daily_below_weekly or price < ema_50_1d_aligned[i] - 2.0 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: daily EMA crosses above weekly EMA OR stop loss hit
            if daily_above_weekly or price > ema_50_1d_aligned[i] + 2.0 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_EMA50_EMA20_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0