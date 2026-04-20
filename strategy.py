#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1-week trend filter, 1-day support/resistance zones, and volume confirmation
# Long when: price above weekly EMA200 AND above 1-day VWAP AND volume > 1.5x 20-day avg
# Short when: price below weekly EMA200 AND below 1-day VWAP AND volume > 1.5x 20-day avg
# Exit when price crosses 1-day VWAP in opposite direction
# Designed to capture strong trends with institutional volume while avoiding chop
# Weekly trend filter reduces false signals in ranging markets
# Volume confirmation ensures institutional participation
# Target: 50-150 trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA200 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Load daily data ONCE for VWAP and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily VWAP calculation
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vp_1d = typical_price_1d * volume_1d
    cum_vp_1d = np.cumsum(vp_1d)
    cum_vol_1d = np.cumsum(volume_1d)
    vwap_1d = np.divide(cum_vp_1d, cum_vol_1d, out=np.full_like(cum_vp_1d, np.nan), where=cum_vol_1d!=0)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Daily volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # 6h price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_200_val = ema_200_1w_aligned[i]
        vwap_val = vwap_1d_aligned[i]
        vol_ma_val = volume_ma_20_1d_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: current volume must be above 1.5x 20-day average
        vol_filter = vol > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: price above weekly EMA200 AND above daily VWAP AND volume confirmation
            if price > ema_200_val and price > vwap_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA200 AND below daily VWap AND volume confirmation
            elif price < ema_200_val and price < vwap_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below daily VWAP
            if price < vwap_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above daily VWAP
            if price > vwap_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1w_EMA200_1d_VWAP_VolumeFilter"
timeframe = "6h"
leverage = 1.0