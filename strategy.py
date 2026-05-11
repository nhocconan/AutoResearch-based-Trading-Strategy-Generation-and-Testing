# 1h_VWAP_Reversion_4hTrend_1dVolFilter
# Hypothesis: Price mean-reverts to VWAP during 4h trends with daily volume confirmation.
# Uses 4h EMA for trend direction, 1d volume MA for filter, and 1h VWAP deviation for entry.
# Designed for low trade frequency (15-35/year) to avoid fee drag while capturing mean reversion in trends.

name = "1h_VWAP_Reversion_4hTrend_1dVolFilter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 4h EMA20 trend ---
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # --- Daily volume MA20 ---
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # --- 1h VWAP (typical price * volume) / cumulative volume ---
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = vwap_num / vwap_den
    # Avoid division by zero at start
    vwap[vwap_den == 0] = close[vwap_den == 0]
    
    # --- VWAP deviation bands (2 std dev of typical price) ---
    # Calculate rolling std of typical price
    tp_series = pd.Series(typical_price)
    tp_std = tp_series.rolling(window=20, min_periods=20).std().values
    vwap_upper = vwap + 2.0 * tp_std
    vwap_lower = vwap - 2.0 * tp_std
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: ensure we have enough data for indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_20_4h_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(vwap[i]) or
            np.isnan(vwap_upper[i]) or
            np.isnan(vwap_lower[i]) or
            np.isnan(tp_std[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Trend filter from 4h EMA20
        bullish_trend = close[i] > ema_20_4h_aligned[i]
        bearish_trend = close[i] < ema_20_4h_aligned[i]
        
        # Volume filter: current volume > 1.5x daily average
        volume_filter = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0 and in_session:
            # Long: price touches VWAP lower band in bullish trend with volume
            if close[i] <= vwap_lower[i] and bullish_trend and volume_filter:
                signals[i] = 0.20
                position = 1
            # Short: price touches VWAP upper band in bearish trend with volume
            elif close[i] >= vwap_upper[i] and bearish_trend and volume_filter:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price reaches VWAP or opposite band
            if close[i] >= vwap[i] or close[i] >= vwap_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price reaches VWAP or opposite band
            if close[i] <= vwap[i] or close[i] <= vwap_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

#!/usr/bin/env python3