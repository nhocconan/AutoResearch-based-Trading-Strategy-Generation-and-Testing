#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h volume-weighted price action with 1d trend filter
# Uses 12h volume-weighted average price (VWAP) deviation as mean reversion signal
# Combined with 1d EMA trend filter to avoid counter-trend trades
# Volume confirmation ensures institutional participation
# Designed for low frequency (12-37 trades/year) to minimize fee drag
# Works in bull/bear as 1d EMA adapts to trend while VWAP captures short-term extremes

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(34) for trend filter
    ema_len = 34
    if len(df_1d) < ema_len:
        return np.zeros(n)
    
    ema_1d = pd.Series(df_1d['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 12h VWAP calculation (typical price * volume)
    typical_price = (high + low + close) / 3.0
    vwap_numerator = pd.Series(typical_price * volume).rolling(window=24, min_periods=24).sum().values  # 24*12h = 12 days
    vwap_denominator = pd.Series(volume).rolling(window=24, min_periods=24).sum().values
    vwap = vwap_numerator / vwap_denominator
    
    # VWAP deviation as z-score (mean reversion signal)
    vwap_dev = (close - vwap) / vwap
    vwap_z = pd.Series(vwap_dev).rolling(window=50, min_periods=50).apply(
        lambda x: (x[-1] - np.mean(x)) / np.std(x) if np.std(x) > 0 else 0, raw=True
    ).values
    
    # Volume confirmation: current volume > 1.8x average volume (24-period)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(100, 50, 24)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_z[i]) or 
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d EMA34
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.8x average
        volume_confirmed = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Enter long: VWAP mean reversion (oversold) + above 1d EMA + volume
            if (vwap_z[i] < -1.5 and 
                above_ema and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: VWAP mean reversion (overbought) + below 1d EMA + volume
            elif (vwap_z[i] > 1.5 and 
                  below_ema and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to VWAP or breaks below 1d EMA
            if vwap_z[i] > -0.5 or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to VWAP or breaks above 1d EMA
            if vwap_z[i] < 0.5 or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_VWAP_MeanReversion_v1"
timeframe = "12h"
leverage = 1.0