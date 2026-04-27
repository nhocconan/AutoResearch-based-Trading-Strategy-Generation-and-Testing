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
    
    # Get 1d data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily average volume for volume filter
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily RSI(14) for mean reversion
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d = rsi_14_1d.values
    
    # Align indicators to lower timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need all indicators
    start_idx = max(34, 20, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i]) or np.isnan(rsi_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        ema_trend = ema_34_1w_aligned[i]
        atr_val = atr_14_1d_aligned[i]
        vol_avg = vol_avg_1d_aligned[i]
        vol_current = volume[i]
        rsi_val = rsi_14_1d_aligned[i]
        
        # Volatility filter: ATR > 20-period median (high volatility regime)
        if i >= 20:
            atr_ma = pd.Series(atr_14_1d_aligned[:i+1]).rolling(window=20, min_periods=20).median().iloc[-1]
        else:
            atr_ma = atr_val
        vol_filter = atr_val > atr_ma
        
        # Volume filter: current volume > 1.5x daily average
        volume_filter = vol_current > (vol_avg * 1.5)
        
        # RSI mean reversion filter: avoid extreme RSI
        rsi_filter = (rsi_val > 20) and (rsi_val < 80)
        
        # Entry conditions: only trade when trend, volatility, volume, and RSI conditions align
        if position == 0:
            # Long: weekly trend up + volatility + volume + RSI not overbought
            if close[i] > ema_trend and vol_filter and volume_filter and rsi_filter:
                signals[i] = size
                position = 1
            # Short: weekly trend down + volatility + volume + RSI not oversold
            elif close[i] < ema_trend and vol_filter and volume_filter and rsi_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend reversal or volatility collapse or RSI overbought
            if close[i] < ema_trend or atr_val < (atr_ma * 0.8) or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: trend reversal or volatility collapse or RSI oversold
            if close[i] > ema_trend or atr_val < (atr_ma * 0.8) or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyTrend_VolumeVolatilityRSIFilter"
timeframe = "6h"
leverage = 1.0