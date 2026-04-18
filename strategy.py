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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 34-period EMA on 1d for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 4h
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h ATR (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h volume spike (volume > 2.5x 30-period average)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    # Calculate 4h ADX (14-period) for trend strength filter
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / tr14
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / tr14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 30, 14, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        vol_confirmed = volume_spike[i]
        
        # Trend strength filter: ADX > 25
        strong_trend = adx[i] > 25
        
        if position == 0:
            # Long: price above 1d EMA34 with volume spike and strong trend
            if uptrend and vol_confirmed and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: price below 1d EMA34 with volume spike and strong trend
            elif downtrend and vol_confirmed and strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 1d EMA34 OR trend weakens
            if close[i] < ema_34_1d_aligned[i] or adx[i] < 20:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 1d EMA34 OR trend weakens
            if close[i] > ema_34_1d_aligned[i] or adx[i] < 20:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dEMA34_VolumeSpike_ADXFilter_v1"
timeframe = "4h"
leverage = 1.0