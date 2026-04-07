#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h mean reversion with 4h trend filter and 1d volume regime
# Long in uptrend when price pulls back to VWAP with high volume
# Short in downtrend when price bounces from VWAP with high volume
# Uses 4h for trend direction (EMA50), 1d for volume regime (high/low volume days)
# Target: 15-35 trades/year, low frequency to minimize fee drag
name = "1h_vwap_mean_reversion_4h_trend_1d_volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d data for volume regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume z-score (regime filter)
    vol_1d = df_1d['volume'].values
    vol_mean = pd.Series(vol_1d).rolling(window=30, min_periods=30).mean().values
    vol_std = pd.Series(vol_1d).rolling(window=30, min_periods=30).std().values
    vol_z = (vol_1d - vol_mean) / np.where(vol_std == 0, 1, vol_std)
    vol_z_aligned = align_htf_to_ltf(prices, df_1d, vol_z)
    
    # Calculate VWAP (typical price * volume cumulative)
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    signals = np.zeros(n)
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(vol_z_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1 = uptrend (price > EMA), -1 = downtrend (price < EMA)
        trend = 1 if close[i] > ema_4h_aligned[i] else -1
        
        # Volume regime: only trade on high volume days (z-score > 0.5)
        vol_regime = vol_z_aligned[i] > 0.5
        
        # Price deviation from VWAP
        vwap_dev = (close[i] - vwap[i]) / vwap[i] if vwap[i] != 0 else 0
        
        # Entry conditions
        if trend == 1 and vol_regime:  # Uptrend + high volume
            # Long when price pulls back to VWAP (negative deviation)
            if vwap_dev < -0.005:  # 0.5% below VWAP
                signals[i] = 0.20
            else:
                signals[i] = 0.0
        elif trend == -1 and vol_regime:  # Downtrend + high volume
            # Short when price bounces from VWAP (positive deviation)
            if vwap_dev > 0.005:  # 0.5% above VWAP
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals