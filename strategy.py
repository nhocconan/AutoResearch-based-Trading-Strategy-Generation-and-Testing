#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day volume-weighted average price (VWAP) as dynamic support/resistance
# with 1-week MACD trend filter and volume confirmation.
# Long when price crosses above 1-day VWAP with MACD bullish and volume > 2x average.
# Short when price crosses below 1-day VWAP with MACD bearish and volume > 2x average.
# Exit when price crosses back below/above VWAP or MACD reverses.
# VWAP adapts to market structure, MACD filters trend direction, volume confirms breakout strength.
# Designed for low trade frequency (target: 20-25 trades/year) to minimize fee drag in bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate typical price and VWAP for 1d
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vp_1d = typical_price_1d * volume_1d
    cum_vp_1d = np.nancumsum(vp_1d)
    cum_vol_1d = np.nancumsum(volume_1d)
    vwap_1d = np.divide(cum_vp_1d, cum_vol_1d, out=np.full_like(cum_vp_1d, np.nan), where=cum_vol_1d!=0)
    
    # Load 1w data ONCE for MACD trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate MACD on 1w
    ema_fast = pd.Series(close_1w).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_slow = pd.Series(close_1w).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    # Align indicators to lower timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    macd_hist_aligned = align_htf_to_ltf(prices, df_1w, macd_hist)
    
    # Volume confirmation: 2x average volume (higher threshold for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 35)  # Need volume MA and MACD
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(macd_hist_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 2.0 * vol_ma[i]
        
        # MACD trend filter: hist > 0 for bullish, < 0 for bearish
        macd_bullish = macd_hist_aligned[i] > 0
        macd_bearish = macd_hist_aligned[i] < 0
        
        if position == 0:
            # Look for VWAP crossovers
            # Long: price crosses above VWAP with bullish MACD
            if (close[i] > vwap_1d_aligned[i] and 
                close[i-1] <= vwap_1d_aligned[i-1] and  # crossed above
                macd_bullish and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price crosses below VWAP with bearish MACD
            elif (close[i] < vwap_1d_aligned[i] and 
                  close[i-1] >= vwap_1d_aligned[i-1] and  # crossed below
                  macd_bearish and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below VWAP or MACD turns bearish
            if (close[i] < vwap_1d_aligned[i] and 
                close[i-1] >= vwap_1d_aligned[i-1]) or \
               macd_hist_aligned[i] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above VWAP or MACD turns bullish
            if (close[i] > vwap_1d_aligned[i] and 
                close[i-1] <= vwap_1d_aligned[i-1]) or \
               macd_hist_aligned[i] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_VWAP_MACD_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0