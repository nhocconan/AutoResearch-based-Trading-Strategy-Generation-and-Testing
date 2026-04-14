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
    open_time = pd.to_datetime(prices['open_time'])
    
    # Pre-compute hour for session filter (08-20 UTC)
    hours = open_time.dt.hour.values
    
    # Load 1w and 1d data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ATR (14-period) for volatility filter
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1w EMA(20) for long-term trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 1d EMA(50) for medium-term trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 14-period RSI for momentum confirmation
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs)).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size (balanced for 12h)
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Session filter: only trade 08-20 UTC
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any critical data is NaN
        if (np.isnan(atr[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Calculate average volume for confirmation (20-period)
        vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().iloc[i]
        
        # ATR-based volatility filter: avoid extremely low volatility periods
        atr_ratio = atr[i] / price if price > 0 else 0
        vol_filter = atr_ratio > 0.005  # Minimum 0.5% ATR relative to price
        
        # Volume confirmation: current volume > 1.5x average (stricter for fewer trades)
        vol_confirm = vol > (vol_avg * 1.5) if not np.isnan(vol_avg) else False
        
        # Trend filter: price > both 1w EMA20 and 1d EMA50 for long, price < both for short
        trend_filter_long = price > ema_20_1w_aligned[i] and price > ema_50_1d_aligned[i]
        trend_filter_short = price < ema_20_1w_aligned[i] and price < ema_50_1d_aligned[i]
        
        # Momentum filter: RSI between 40 and 60 to avoid extremes (tighter range)
        rsi_filter = 40 <= rsi[i] <= 60
        
        if position == 0:
            # Long setup: price above both EMAs + volume confirmation + volatility filter + RSI filter
            if (trend_filter_long and vol_confirm and vol_filter and rsi_filter):
                position = 1
                signals[i] = position_size
            # Short setup: price below both EMAs + volume confirmation + volatility filter + RSI filter
            elif (trend_filter_short and vol_confirm and vol_filter and rsi_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below either EMA
            if price < ema_20_1w_aligned[i] or price < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above either EMA
            if price > ema_20_1w_aligned[i] or price > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w1dEMA_Volume_RSI_Filter"
timeframe = "12h"
leverage = 1.0