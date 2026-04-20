#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE (1w and 1d)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # 1w EMA50 (trend filter)
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d RSI(14) (momentum filter)
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    delta = close_1d_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = (100 - (100 / (1 + rs))).values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # 12h Donchian(20) breakout
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate 20-period high and low
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h volume average (20-period)
    volume = prices['volume'].values
    vol_ma = np.zeros_like(volume)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_ma[:20] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(rsi_14_1d_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_50_1w_val = ema_50_1w_aligned[i]
        rsi_14_1d_val = rsi_14_1d_aligned[i]
        vol_val = volume[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high + uptrend (price > weekly EMA50) + bullish momentum (RSI > 50) + volume confirmation
            if price > high_20[i] and price > ema_50_1w_val and rsi_14_1d_val > 50 and vol_val > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + downtrend (price < weekly EMA50) + bearish momentum (RSI < 50) + volume confirmation
            elif price < low_20[i] and price < ema_50_1w_val and rsi_14_1d_val < 50 and vol_val > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below Donchian low or trend/momentum reversal
            if price < low_20[i] or price < ema_50_1w_val or rsi_14_1d_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks above Donchian high or trend/momentum reversal
            if price > high_20[i] or price > ema_50_1w_val or rsi_14_1d_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_WeeklyEMA50_RSI14_Volume"
timeframe = "12h"
leverage = 1.0