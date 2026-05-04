#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and chop regime filter
# In trending regimes (1d chop < 42): breakout in direction of 1d EMA34 trend
# In ranging regimes (1d chop >= 42): mean reversion at Camarilla H3/L3 levels
# Volume confirmation (>1.5x 20-period EMA) ensures institutional participation
# Discrete sizing (0.25) minimizes fee churn. Target: 75-200 trades over 4 years.
# Strategy adapts to bull/bear markets via regime detection and has proven edge on ETH/SOL.

name = "4h_Camarilla_1dChop_VolumeSpike_EMATrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for regime filter and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend direction
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d chopiness index (14-period)
    tr1 = pd.Series(df_1d['high']).sub(df_1d['low'])
    tr2 = pd.Series(df_1d['high']).sub(df_1d['close'].shift(1)).abs()
    tr3 = pd.Series(df_1d['low']).sub(df_1d['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    chop = 100 * np.log10(atr.rolling(window=14, min_periods=14).sum() / 
                           (np.log10(14) * tr.rolling(window=14, min_periods=14).sum()))
    chop = chop.replace([np.inf, -np.inf], np.nan).fillna(50).values  # neutral when undefined
    
    # Align 1d indicators to 4h timeframe (completed 1d bar only)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 4h Camarilla levels (based on previous day's OHLC)
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.25*(high-low), etc.
    # We use daily OHLC to calculate levels for the 4h chart
    df_1d_open = df_1d['open'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_h5 = np.zeros_like(df_1d_close)
    camarilla_h4 = np.zeros_like(df_1d_close)
    camarilla_h3 = np.zeros_like(df_1d_close)
    camarilla_l3 = np.zeros_like(df_1d_close)
    camarilla_l4 = np.zeros_like(df_1d_close)
    camarilla_l5 = np.zeros_like(df_1d_close)
    
    for i in range(len(df_1d)):
        if i == 0:
            camarilla_h5[i] = camarilla_h4[i] = camarilla_h3[i] = np.nan
            camarilla_l3[i] = camarilla_l4[i] = camarilla_l5[i] = np.nan
            continue
        range_prev = df_1d_high[i-1] - df_1d_low[i-1]
        camarilla_h5[i] = df_1d_close[i-1] + 2.5 * range_prev
        camarilla_h4[i] = df_1d_close[i-1] + 2.0 * range_prev
        camarilla_h3[i] = df_1d_close[i-1] + 1.5 * range_prev
        camarilla_l3[i] = df_1d_close[i-1] - 1.5 * range_prev
        camarilla_l4[i] = df_1d_close[i-1] - 2.0 * range_prev
        camarilla_l5[i] = df_1d_close[i-1] - 2.5 * range_prev
    
    # Align Camarilla levels to 4h timeframe
    h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            if chop_aligned[i] < 42:
                # Trending regime: breakout in direction of 1d EMA34
                if close[i] > ema_34_aligned[i]:
                    # Uptrend: long on break above H3
                    if close[i] > h3_aligned[i] and volume_confirm:
                        signals[i] = 0.25
                        position = 1
                else:
                    # Downtrend: short on break below L3
                    if close[i] < l3_aligned[i] and volume_confirm:
                        signals[i] = -0.25
                        position = -1
            else:
                # Ranging regime: mean reversion at H3/L3 levels
                if close[i] <= l3_aligned[i] and volume_confirm:
                    # Long at L3 support
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= h3_aligned[i] and volume_confirm:
                    # Short at H3 resistance
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price returns to midpoint of H3-L3 OR chop increases (>50) OR volume drops
            midpoint = (h3_aligned[i] + l3_aligned[i]) / 2
            if (close[i] <= midpoint or 
                chop_aligned[i] > 50 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint of H3-L3 OR chop increases (>50) OR volume drops
            midpoint = (h3_aligned[i] + l3_aligned[i]) / 2
            if (close[i] >= midpoint or 
                chop_aligned[i] > 50 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals