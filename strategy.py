#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using daily volume-weighted average price (VWAP) as dynamic support/resistance
# VWAP acts as institutional reference price - price tends to revert to VWAP in ranging markets
# Breakouts above/below VWAP with volume confirmation indicate institutional participation
# Trend filter: 20-period EMA on 6x to align with short-term momentum
# Works in bull/bear markets: VWAP reversion in ranges, VWAP breakouts in trends
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_VWAP_Reversion_Breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily VWAP ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Typical price for VWAP calculation
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # VWAP = cumulative(typical_price * volume) / cumulative(volume)
    vwap = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    
    # Align daily VWAP to 6h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap.values)
    
    # Volume confirmation: >1.8x 20-period average to filter weak moves
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma_20)
    
    # Trend filter: 20-period EMA on 6h timeframe
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    uptrend = close > ema_20
    downtrend = close < ema_20
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(vwap_aligned[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(ema_20[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above VWAP with volume confirmation and uptrend
            if close[i] > vwap_aligned[i] and volume_filter[i] and uptrend[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below VWAP with volume confirmation and downtrend
            elif close[i] < vwap_aligned[i] and volume_filter[i] and downtrend[i]:
                signals[i] = -0.25
                position = -1
            # Long reversion: price pulls back to VWAP from above with volume
            elif close[i] > vwap_aligned[i] and abs(close[i] - vwap_aligned[i]) / vwap_aligned[i] < 0.003 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short reversion: price pulls back to VWAP from below with volume
            elif close[i] < vwap_aligned[i] and abs(close[i] - vwap_aligned[i]) / vwap_aligned[i] < 0.003 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below VWAP (failed breakout) or reverts to VWAP (take profit)
            if close[i] < vwap_aligned[i] or abs(close[i] - vwap_aligned[i]) / vwap_aligned[i] < 0.002:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above VWAP (failed breakdown) or reverts to VWAP (take profit)
            if close[i] > vwap_aligned[i] or abs(close[i] - vwap_aligned[i]) / vwap_aligned[i] < 0.002:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals