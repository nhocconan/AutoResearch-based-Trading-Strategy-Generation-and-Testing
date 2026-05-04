#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted RSI + 12h Supertrend Combination
# Uses Volume-Weighted RSI (VW-RSI) on 6h to identify overextended conditions with volume confirmation,
# combined with 12h Supertrend to filter trades in the direction of the higher-timeframe trend.
# VW-RSI incorporates volume into RSI calculation, giving more weight to price moves with higher volume.
# Supertrend (ATR=10, mult=3.0) provides objective trend direction and dynamic support/resistance.
# Designed for 12-35 trades/year (~50-140 total over 4 years) with discrete position sizing to minimize fee drag.
# Works in bull markets (long when VW-RSI oversold in uptrend) and bear markets (short when VW-RSI overbought in downtrend).

name = "6h_VolWeightedRSI_12hSupertrend_Combo"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Supertrend calculation - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ATR for Supertrend (period=10)
    tr1_12h = np.abs(high_12h[1:] - low_12h[:-1])
    tr2_12h = np.abs(high_12h[1:] - close_12h[:-1])
    tr3_12h = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h = np.concatenate([[np.nan], tr_12h])  # align length
    
    atr_12h = pd.Series(tr_12h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate Supertrend basic upper and lower bands
    hl2_12h = (high_12h + low_12h) / 2
    upper_basic_12h = hl2_12h + (3.0 * atr_12h)
    lower_basic_12h = hl2_12h - (3.0 * atr_12h)
    
    # Initialize Supertrend components
    upper_band_12h = np.full_like(close_12h, np.nan)
    lower_band_12h = np.full_like(close_12h, np.nan)
    supertrend_12h = np.full_like(close_12h, np.nan)
    trend_12h = np.ones_like(close_12h, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    # Calculate Supertrend iteratively
    for i in range(1, len(close_12h)):
        if np.isnan(atr_12h[i]) or np.isnan(upper_basic_12h[i]) or np.isnan(lower_basic_12h[i]):
            continue
            
        # Upper band
        if i == 1:
            upper_band_12h[i] = upper_basic_12h[i]
        else:
            if close_12h[i-1] <= upper_band_12h[i-1]:
                upper_band_12h[i] = min(upper_basic_12h[i], upper_band_12h[i-1])
            else:
                upper_band_12h[i] = upper_basic_12h[i]
        
        # Lower band
        if i == 1:
            lower_band_12h[i] = lower_basic_12h[i]
        else:
            if close_12h[i-1] >= lower_band_12h[i-1]:
                lower_band_12h[i] = max(lower_basic_12h[i], lower_band_12h[i-1])
            else:
                lower_band_12h[i] = lower_basic_12h[i]
        
        # Supertrend and trend
        if i == 1:
            supertrend_12h[i] = upper_band_12h[i]
            trend_12h[i] = 1
        else:
            if trend_12h[i-1] == 1 and close_12h[i] <= upper_band_12h[i]:
                trend_12h[i] = -1
                supertrend_12h[i] = lower_band_12h[i]
            elif trend_12h[i-1] == -1 and close_12h[i] >= lower_band_12h[i]:
                trend_12h[i] = 1
                supertrend_12h[i] = upper_band_12h[i]
            else:
                trend_12h[i] = trend_12h[i-1]
                supertrend_12h[i] = upper_band_12h[i] if trend_12h[i] == 1 else lower_band_12h[i]
    
    # Align Supertrend to 6h timeframe (wait for completed 12h bar)
    supertrend_12h_aligned = align_htf_to_ltf(prices, df_12h, supertrend_12h)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h.astype(float))
    
    # Calculate Volume-Weighted RSI on 6h (period=14)
    # VW-RSI = 100 - (100 / (1 + RS)), where RS = Average Gain / Average Loss
    # Gains and Losses are volume-weighted
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Volume-weight the gains and losses
    vol_gain = gain * volume
    vol_loss = loss * volume
    
    # Calculate average volume-weighted gain and loss (Wilder's smoothing)
    avg_vol_gain = pd.Series(vol_gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_vol_loss = pd.Series(vol_loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Avoid division by zero
    rs = np.where(avg_vol_loss != 0, avg_vol_gain / avg_vol_loss, 0)
    vw_rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(supertrend_12h_aligned[i]) or np.isnan(trend_12h_aligned[i]) or 
            np.isnan(vw_rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vw_rsi_val = vw_rsi[i]
        supertrend_val = supertrend_12h_aligned[i]
        trend_val = trend_12h_aligned[i]
        
        if position == 0:
            # Long conditions: VW-RSI oversold (<30) AND 12h trend is up (1)
            if vw_rsi_val < 30 and trend_val == 1:
                signals[i] = 0.25
                position = 1
            # Short conditions: VW-RSI overbought (>70) AND 12h trend is down (-1)
            elif vw_rsi_val > 70 and trend_val == -1:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: VW-RSI overbought (>70) OR price crosses below Supertrend
            if vw_rsi_val > 70 or close[i] < supertrend_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: VW-RSI oversold (<30) OR price crosses above Supertrend
            if vw_rsi_val < 30 or close[i] > supertrend_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals