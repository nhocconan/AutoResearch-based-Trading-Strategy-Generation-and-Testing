#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Supertrend with 12h EMA filter and volume confirmation
# Uses Supertrend(ATR=10, mult=3) for trend direction on 6m timeframe.
# Filters trades with 12h EMA(20) to avoid counter-trend entries.
# Requires volume > 1.5x 20-period average for confirmation.
# Designed for moderate trade frequency (target: 15-35 trades/year) with strong trend filtering.
# Works in bull markets via trend following and in bear markets via short signals.
# Supertrend adapts to volatility, reducing whipsaw in ranging markets.

name = "6h_supertrend_12h_ema_volume_v1"
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
    
    # 12h data for EMA filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate ATR(10) for Supertrend
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high + low) / 2
    upper_band = hl2 + 3.0 * atr
    lower_band = hl2 - 3.0 * atr
    
    upper_band_final = np.zeros(n)
    lower_band_final = np.zeros(n)
    upper_band_final[0] = upper_band[0]
    lower_band_final[0] = lower_band[0]
    
    for i in range(1, n):
        if close[i-1] <= upper_band_final[i-1]:
            upper_band_final[i] = min(upper_band[i], upper_band_final[i-1])
        else:
            upper_band_final[i] = upper_band[i]
            
        if close[i-1] >= lower_band_final[i-1]:
            lower_band_final[i] = max(lower_band[i], lower_band_final[i-1])
        else:
            lower_band_final[i] = lower_band[i]
    
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 for uptrend, -1 for downtrend
    supertrend[0] = lower_band_final[0]
    trend[0] = 1
    
    for i in range(1, n):
        if close[i] > upper_band_final[i-1]:
            trend[i] = 1
        elif close[i] < lower_band_final[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            
        if trend[i] == 1:
            supertrend[i] = lower_band_final[i]
        else:
            supertrend[i] = upper_band_final[i]
    
    # 12h EMA(20) for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(supertrend[i]) or np.isnan(trend[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Supertrend signal
        bullish = trend[i] == 1
        bearish = trend[i] == -1
        
        # 12h EMA filter: only trade in direction of higher timeframe trend
        ema_filter_long = close[i] > ema_12h_aligned[i]
        ema_filter_short = close[i] < ema_12h_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Long: Supertrend uptrend + price above 12h EMA + volume confirmation
        if bullish and ema_filter_long and vol_confirm:
            signals[i] = 0.25
        # Short: Supertrend downtrend + price below 12h EMA + volume confirmation
        elif bearish and ema_filter_short and vol_confirm:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals