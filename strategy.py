#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with weekly pivot breakouts (R2/S2) filtered by weekly volume and trend.
# Weekly pivot levels provide stronger support/resistance than daily. Volume confirmation ensures
# institutional interest. Trend filter (EMA34) avoids counter-trend trades. Designed for 15-25 trades/year.
# Works in bull (breakouts continue) and bear (fades at S2/R2) due to mean-reversion at extremes.

name = "12h_WeeklyPivot_R2S2_Breakout_VolumeTrend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly high, low, close for Camarilla pivot calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Calculate R2 and S2 using Camarilla formula
    r2_1w = close_1w + (high_1w - low_1w) * 1.1 / 6
    s2_1w = close_1w - (high_1w - low_1w) * 1.1 / 6
    
    # Align weekly pivot levels to 12h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Weekly ATR for volatility filter (14-period)
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.absolute(high_1w[1:] - close_1w[:-1]))
    tr1 = np.maximum(tr1, np.absolute(low_1w[1:] - close_1w[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    atr_14_1w = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average (12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Trend filter: price above/below 34-period EMA
    close_series = pd.Series(close)
    ema_34 = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 80
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or 
            np.isnan(s2_1w_aligned[i]) or np.isnan(atr_14_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema_34[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        pivot = pivot_1w_aligned[i]
        r2 = r2_1w_aligned[i]
        s2 = s2_1w_aligned[i]
        atr = atr_14_1w_aligned[i]
        ema = ema_34[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: break above R2 with volume and above EMA
            if price > r2 and volume_confirmed and price > ema:
                signals[i] = 0.25
                position = 1
            # Short: break below S2 with volume and below EMA
            elif price < s2 and volume_confirmed and price < ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below pivot or EMA
            if price < pivot or price < ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above pivot or EMA
            if price > pivot or price > ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals