#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h RSI for momentum direction and 1d VWAP for mean reversion.
# In trending markets (4h RSI > 50), go long on pullbacks to 1d VWAP.
# In ranging markets (4h RSI between 40-60), fade extremes of 1d VWAP bands.
# Volume filter confirms institutional participation.
# Target: 15-30 trades/year per symbol to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for RSI momentum filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # RSI(14) on 4h close
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Get 1d data for VWAP and standard deviation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    volume_1d = df_1d['volume'].values
    # VWAP calculation
    vwap_num = np.cumsum(typical_price_1d * volume_1d)
    vwap_den = np.cumsum(volume_1d)
    vwap_1d = vwap_num / (vwap_den + 1e-10)
    # Standard deviation of price from VWAP
    dev = typical_price_1d - vwap_1d
    vwap_var = pd.Series(dev * dev).ewm(span=20, adjust=False, min_periods=20).mean().values
    vwap_std = np.sqrt(vwap_var)
    # VWAP bands: ±1.5 standard deviations
    vwap_upper_1d = vwap_1d + 1.5 * vwap_std
    vwap_lower_1d = vwap_1d - 1.5 * vwap_std
    # Align to 1h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    vwap_upper_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_upper_1d)
    vwap_lower_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_lower_1d)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(vwap_upper_1d_aligned[i]) or np.isnan(vwap_lower_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        # Trending market: 4h RSI > 50 (bullish) or < 50 (bearish)
        if rsi_4h_aligned[i] > 50:  # Bullish trend
            # Long on pullback to VWAP
            if close[i] <= vwap_1d_aligned[i] * 1.005 and close[i] >= vwap_1d_aligned[i] * 0.995:
                if volume_filter[i]:
                    signals[i] = 0.20
                    position = 1
        elif rsi_4h_aligned[i] < 50:  # Bearish trend
            # Short on rally to VWAP
            if close[i] >= vwap_1d_aligned[i] * 0.995 and close[i] <= vwap_1d_aligned[i] * 1.005:
                if volume_filter[i]:
                    signals[i] = -0.20
                    position = -1
        else:  # Ranging market (4h RSI between 40-60)
            # Fade VWAP extremes
            if close[i] <= vwap_lower_1d_aligned[i]:  # Near lower band -> long
                if volume_filter[i]:
                    signals[i] = 0.20
                    position = 1
            elif close[i] >= vwap_upper_1d_aligned[i]:  # Near upper band -> short
                if volume_filter[i]:
                    signals[i] = -0.20
                    position = -1
            else:
                # Hold current position
                if position == 1:
                    signals[i] = 0.20
                elif position == -1:
                    signals[i] = -0.20
                else:
                    signals[i] = 0.0
    
    return signals

name = "1h_RSI4H_VWAP_MeanReversion_Trend"
timeframe = "1h"
leverage = 1.0