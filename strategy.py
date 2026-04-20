#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Chaikin_Momentum_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # === 1d: Calculate Chaikin Money Flow (20) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Money Flow Multiplier
    mfm = ((close_1d - low_1d) - (high_1d - close_1d)) / (high_1d - low_1d)
    mfm = np.where((high_1d - low_1d) != 0, mfm, 0)
    
    # Money Flow Volume
    mfv = mfm * volume_1d
    
    # Chaikin Money Flow (20-period sum)
    mfv_series = pd.Series(mfv)
    volume_series = pd.Series(volume_1d)
    cmf = (mfv_series.rolling(window=20, min_periods=20).sum() / 
           volume_series.rolling(window=20, min_periods=20).sum()).values
    
    # === 1d: Calculate 20-period EMA for trend ===
    close_series = pd.Series(close_1d)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === 1w: Calculate RSI(14) for regime filter ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    delta = pd.Series(close_1w).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1w = (100 - (100 / (1 + rs))).values
    
    # Align all indicators to 1d timeframe
    cmf_aligned = align_htf_to_ltf(prices, df_1d, cmf)
    ema20_aligned = align_htf_to_ltf(prices, df_1d, ema20)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # === 1d: Price and volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        close_val = close[i]
        cmf_val = cmf_aligned[i]
        ema20_val = ema20_aligned[i]
        rsi_1w_val = rsi_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(cmf_val) or np.isnan(ema20_val) or np.isnan(rsi_1w_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: CMF > 0.1 (accumulation), price above EMA20, weekly RSI > 50 (bullish bias), volume confirmation
            if cmf_val > 0.1 and close_val > ema20_val and rsi_1w_val > 50 and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: CMF < -0.1 (distribution), price below EMA20, weekly RSI < 50 (bearish bias), volume confirmation
            elif cmf_val < -0.1 and close_val < ema20_val and rsi_1w_val < 50 and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: CMF turns negative OR price breaks below EMA20 OR low volume
            if cmf_val < 0 or close_val < ema20_val or vol_ratio_val < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: CMF turns positive OR price breaks above EMA20 OR low volume
            if cmf_val > 0 or close_val > ema20_val or vol_ratio_val < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals