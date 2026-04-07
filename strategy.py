#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Daily Volume-Weighted Average Price (VWAP) Trend with 12h Filter
# Hypothesis: VWAP acts as a dynamic support/resistance level. Price above VWAP indicates bullish
# sentiment, below indicates bearish. Using 12h trend filter ensures we only trade in the direction
# of higher timeframe momentum, reducing whipsaws. Works in both bull/bear markets: in bull markets
# we buy dips to VWAP in uptrends, in bear markets we sell rallies to VWAP in downtrends.
# Volume confirmation adds conviction to the signal.
# Target: 20-40 trades/year (80-160 over 4 years).

name = "6h_vwap_trend_12h_filter_v1"
timeframe = "6h"
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
    
    # Calculate VWAP for 6h
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(volume)
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA(20) for trend
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(vwap[i]) or np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below VWAP or trend turns bearish
            if close[i] < vwap[i] or ema_12h_aligned[i] > close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price crosses above VWAP or trend turns bullish
            if close[i] > vwap[i] or ema_12h_aligned[i] < close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price above VWAP with bullish 12h trend and volume
            if close[i] > vwap[i] and ema_12h_aligned[i] < close[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short: price below VWAP with bearish 12h trend and volume
            elif close[i] < vwap[i] and ema_12h_aligned[i] > close[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals