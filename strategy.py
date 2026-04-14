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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily 20-period EMA for trend
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate daily 20-period SMA for Donchian channels
    sma_20_1d_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).mean().values
    sma_20_1d_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).mean().values
    sma_20_1d_high_aligned = align_htf_to_ltf(prices, df_1d, sma_20_1d_high)
    sma_20_1d_low_aligned = align_htf_to_ltf(prices, df_1d, sma_20_1d_low)
    
    # Calculate daily ATR(14) for volatility filter
    high_series = pd.Series(df_1d['high'])
    low_series = pd.Series(df_1d['low'])
    close_series = pd.Series(df_1d['close'])
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = tr.rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily average volume for confirmation
    vol_avg_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_20_1d_aligned[i]) or 
            np.isnan(sma_20_1d_high_aligned[i]) or 
            np.isnan(sma_20_1d_low_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volatility filter: avoid extremely low volatility periods
        atr_ratio = atr_14_1d_aligned[i] / price if price > 0 else 0
        vol_filter = atr_ratio > 0.003  # Minimum 0.3% ATR relative to price
        
        # Volume confirmation: current volume > 1.5x daily average
        vol_confirm = vol > (vol_avg_1d_aligned[i] * 1.5) if not np.isnan(vol_avg_1d_aligned[i]) else False
        
        # Trend filter: price above/below daily EMA20
        trend_filter_long = price > ema_20_1d_aligned[i]
        trend_filter_short = price < ema_20_1d_aligned[i]
        
        # Donchian breakout conditions
        donchian_breakout_long = price > sma_20_1d_high_aligned[i]
        donchian_breakout_short = price < sma_20_1d_low_aligned[i]
        
        if position == 0:
            # Long setup: Donchian breakout + trend alignment + volume + volatility
            if (donchian_breakout_long and trend_filter_long and vol_confirm and vol_filter):
                position = 1
                signals[i] = position_size
            # Short setup: Donchian breakdown + trend alignment + volume + volatility
            elif (donchian_breakout_short and trend_filter_short and vol_confirm and vol_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below lower Donchian band
            if price < sma_20_1d_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above upper Donchian band
            if price > sma_20_1d_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_DailyDonchian20_EMA20_Volume"
timeframe = "12h"
leverage = 1.0