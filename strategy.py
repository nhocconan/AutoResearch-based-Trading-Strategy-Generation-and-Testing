#!/usr/bin/env python3
# mtf_1h_supertrend_ema_volume_v1
# Hypothesis: 1h strategy using 4h Supertrend for trend direction and 1d EMA200 for institutional bias.
# Enters on pullbacks to EMA21 with volume confirmation during London/NY session (08-20 UTC).
# Works in bull/bear: 4h Supertrend catches major trends, 1d EMA200 filters counter-trend noise.
# Target: 15-35 trades/year by requiring confluence of 4h trend, 1d bias, EMA pullback, volume, and session.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_supertrend_ema_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    
    # 4h HTF data for Supertrend trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h ATR(10) for Supertrend
    tr_4h = np.maximum(np.maximum(high_4h[1:] - low_4h[1:], np.abs(high_4h[1:] - close_4h[:-1])), np.abs(low_4h[1:] - close_4h[:-1]))
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr_4h = pd.Series(tr_4h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # 4h Supertrend calculation
    hl2_4h = (high_4h + low_4h) / 2
    upper_4h = hl2_4h + 3.0 * atr_4h
    lower_4h = hl2_4h - 3.0 * atr_4h
    
    upper_4h[0] = np.nan
    lower_4h[0] = np.nan
    
    for i in range(1, len(close_4h)):
        if close_4h[i-1] <= upper_4h[i-1]:
            upper_4h[i] = min(upper_4h[i], upper_4h[i-1])
        else:
            upper_4h[i] = upper_4h[i]
        if close_4h[i-1] >= lower_4h[i-1]:
            lower_4h[i] = max(lower_4h[i], lower_4h[i-1])
        else:
            lower_4h[i] = lower_4h[i]
    
    supertrend_4h = np.full(len(close_4h), np.nan)
    for i in range(len(close_4h)):
        if np.isnan(upper_4h[i]) or np.isnan(lower_4h[i]):
            supertrend_4h[i] = np.nan
        elif i == 0:
            supertrend_4h[i] = np.nan
        elif supertrend_4h[i-1] == upper_4h[i-1]:
            supertrend_4h[i] = lower_4h[i] if close_4h[i] > upper_4h[i] else upper_4h[i]
        else:
            supertrend_4h[i] = upper_4h[i] if close_4h[i] < lower_4h[i] else lower_4h[i]
    
    # Align 4h Supertrend to 1h (trend direction: 1=uptrend, -1=downtrend)
    supertrend_dir_4h = np.where(close_4h > supertrend_4h, 1, -1)
    supertrend_dir_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend_dir_4h)
    
    # 1d HTF data for EMA200 institutional bias
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 1h EMA21 for dynamic support/resistance
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 1h volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_dir_4h_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or
            np.isnan(ema21[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                # Exit outside session
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: trend turns bearish OR price breaks below EMA21
            if supertrend_dir_4h_aligned[i] == -1 or close[i] < ema21[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: trend turns bullish OR price breaks above EMA21
            if supertrend_dir_4h_aligned[i] == 1 or close[i] > ema21[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long: 4h uptrend, price above 1d EMA200, pullback to EMA21
                if (supertrend_dir_4h_aligned[i] == 1 and 
                    close[i] > ema200_1d_aligned[i] and 
                    close[i] <= ema21[i] * 1.005):  # Within 0.5% above EMA21 (pullback)
                    position = 1
                    signals[i] = 0.20
                # Short: 4h downtrend, price below 1d EMA200, pullback to EMA21
                elif (supertrend_dir_4h_aligned[i] == -1 and 
                      close[i] < ema200_1d_aligned[i] and 
                      close[i] >= ema21[i] * 0.995):  # Within 0.5% below EMA21 (pullback)
                    position = -1
                    signals[i] = -0.20
    
    return signals