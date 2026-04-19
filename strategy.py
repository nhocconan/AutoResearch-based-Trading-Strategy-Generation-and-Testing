#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_VWAP_Breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 300:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for VWAP and trend context
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # 1d VWAP calculation (typical price * volume / cumulative volume)
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    vp_1d = typical_price_1d * df_1d['volume'].values
    cum_vp_1d = np.cumsum(vp_1d)
    cum_vol_1d = np.cumsum(df_1d['volume'].values)
    vwap_1d = np.divide(cum_vp_1d, cum_vol_1d, out=np.full_like(cum_vp_1d, np.nan), where=cum_vol_1d!=0)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # 1d ATR for volatility filter and dynamic thresholds
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr_1d = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)), np.absolute(low_1d - np.roll(close_1d, 1)))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1w trend filter: price relative to weekly EMA50
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 6h VWAP deviation: current price vs 1d VWAP (normalized by ATR)
    vwap_dev = (close - vwap_1d_aligned) / atr_1d_aligned
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 14)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if np.isnan(vwap_dev[i]) or np.isnan(atr_1d_aligned[i]) or \
           np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        dev = vwap_dev[i]
        atr = atr_1d_aligned[i]
        weekly_ema = ema50_1w_aligned[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        # Trend filter: align with weekly trend
        uptrend = price > weekly_ema
        downtrend = price < weekly_ema
        
        if position == 0:
            # Long: price breaks above VWAP with volume in uptrend
            if dev > 0.8 and volume_ok and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below VWAP with volume in downtrend
            elif dev < -0.8 and volume_ok and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to VWAP or trend breaks
            if dev < 0.2 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to VWAP or trend breaks
            if dev > -0.2 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals