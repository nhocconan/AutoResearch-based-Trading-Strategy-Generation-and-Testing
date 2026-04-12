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
    
    # Get 1d data for trend and momentum filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d RSI(14)
    close_1d = pd.Series(df_1d['close'].values)
    delta = close_1d.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi14 = 100 - (100 / (1 + rs))
    rsi14_values = rsi14.values
    rsi14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi14_values)
    
    # Calculate 1d 14-period ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14_1d = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        atr14_1d[i] = np.nanmean(tr[i-13:i+1])
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Calculate 4-period RSI for entry timing
    delta_4h = pd.Series(close).diff()
    gain_4h = delta_4h.clip(lower=0)
    loss_4h = -delta_4h.clip(upper=0)
    avg_gain_4h = gain_4h.ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    avg_loss_4h = loss_4h.ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    rs_4h = avg_gain_4h / avg_loss_4h
    rsi4 = 100 - (100 / (1 + rs_4h))
    rsi4_values = rsi4.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(rsi14_1d_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(rsi4_values[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d RSI > 50 for long, < 50 for short
        rsi_long_filter = rsi14_1d_aligned[i] > 50
        rsi_short_filter = rsi14_1d_aligned[i] < 50
        
        # Volatility filter: current 4h volatility > 1d ATR (avoid low volatility chop)
        # Approximate 4h volatility using price range
        vol_4h = (high[i] - low[i]) / close[i] * 100  # percentage range
        vol_filter = vol_4h > (atr14_1d_aligned[i] / close[i] * 100) * 0.5
        
        # Entry timing: 4h RSI extremes for mean reversion within trend
        rsi4_oversold = rsi4_values[i] < 30
        rsi4_overbought = rsi4_values[i] > 70
        
        long_entry = rsi_long_filter and rsi4_oversold and vol_filter
        short_entry = rsi_short_filter and rsi4_overbought and vol_filter
        
        # Exit: RSI mean reversion or trend change
        long_exit = (rsi4_values[i] > 50) or (rsi14_1d_aligned[i] < 45)
        short_exit = (rsi4_values[i] < 50) or (rsi14_1d_aligned[i] > 55)
        
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

name = "4h_1d_rsi_trend_mean_reversion_v1"
timeframe = "4h"
leverage = 1.0