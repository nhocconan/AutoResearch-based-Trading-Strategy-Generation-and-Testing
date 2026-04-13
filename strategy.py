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
    
    # Get 1d data for HTF calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 14-period ATR on 1d for volatility
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    low_close = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        atr_14[i] = np.mean(tr[i-14:i])
    
    # Calculate 20-period EMA on 1d (trend filter)
    close_1d_series = pd.Series(close_1d)
    ema_20_1d = close_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate volume ratio (current vs 20-period average)
    volume_1d_series = pd.Series(volume_1d)
    vol_ma_20 = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume_1d, vol_ma_20, out=np.full_like(volume_1d, np.nan), where=vol_ma_20!=0)
    
    # Align indicators to 6h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(atr_14_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA20
        above_ema = close[i] > ema_20_aligned[i]
        below_ema = close[i] < ema_20_aligned[i]
        
        # Volatility filter: ATR > 0.5 * price (avoid choppy markets)
        vol_filter = atr_14_aligned[i] > 0.005 * close[i]
        
        # Volume filter: volume > 1.5 * average
        vol_confirm = vol_ratio_aligned[i] > 1.5
        
        # Entry conditions: trend + volatility + volume
        long_entry = above_ema and vol_filter and vol_confirm
        short_entry = below_ema and vol_filter and vol_confirm
        
        # Exit conditions: trend reversal or volatility collapse
        exit_long = position == 1 and (below_ema or not vol_filter)
        exit_short = position == -1 and (above_ema or not vol_filter)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_ema20_vol_filter"
timeframe = "6h"
leverage = 1.0