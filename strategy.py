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
    
    # Get 1d data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA20 for trend filter
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate 1d ATR14 for volatility filter
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.maximum(np.abs(high_1d[1:] - close_1d[:-1]),
                                  np.abs(low_1d[1:] - close_1d[:-1])))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Calculate 1d volume MA20 for volume confirmation
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(vol_ma20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1d trend
        bullish_trend = close[i] > ema20_1d_aligned[i]
        bearish_trend = close[i] < ema20_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = volume[i] > (1.5 * vol_ma20_1d_aligned[i])
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr14_1d_aligned[i] > 0.01 * close[i]  # ATR > 1% of price
        
        # Entry logic: trend + volume + volatility filter
        long_entry = bullish_trend and volume_confirm and vol_filter
        short_entry = bearish_trend and volume_confirm and vol_filter
        
        # Exit logic: trend reversal
        long_exit = bearish_trend
        short_exit = bullish_trend
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_ema20_trend_volume_filter_v1"
timeframe = "1d"
leverage = 1.0