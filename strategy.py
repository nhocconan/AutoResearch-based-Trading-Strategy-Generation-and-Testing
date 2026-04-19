#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining weekly pivot levels with volume confirmation and ADX trend filter.
# Uses weekly pivot points (R1, S1) as dynamic support/resistance. Enters long when price breaks above R1
# with volume confirmation and strong trend (ADX > 25). Enters short when price breaks below S1 with
# same conditions. Exits when price returns to the weekly pivot level (PP) or trend weakens.
# Weekly pivot provides institutional reference points that work in both bull and bear markets.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years).
name = "6h_WeeklyPivot_R1S1_Breakout_Volume_ADX"
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
    
    # Get weekly data for pivot points
    df_w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: PP = (H + L + C) / 3
    # R1 = 2*PP - L, S1 = 2*PP - H
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    pp_w = (high_w + low_w + close_w) / 3.0
    r1_w = 2 * pp_w - low_w
    s1_w = 2 * pp_w - high_w
    
    # Align weekly pivots to 6h (wait for weekly bar to close)
    pp_w_aligned = align_htf_to_ltf(prices, df_w, pp_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_w, s1_w)
    
    # ADX calculation (14-period) on 6h data
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
                
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i-1]), 
                       abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        plus_dm_smooth = np.zeros_like(plus_dm)
        minus_dm_smooth = np.zeros_like(minus_dm)
        
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_smooth[period] = np.mean(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.mean(minus_dm[1:period+1])
        
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        dx = np.zeros_like(close)
        dx[period:] = 100 * np.abs(plus_di[period:] - minus_di[period:]) / (plus_di[period:] + minus_di[period:])
        
        adx = np.zeros_like(close)
        adx[2*period] = np.mean(dx[period:2*period+1])
        for i in range(2*period+1, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 28)  # Ensure volume MA and ADX are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_w_aligned[i]) or np.isnan(r1_w_aligned[i]) or 
            np.isnan(s1_w_aligned[i]) or np.isnan(adx[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        pp = pp_w_aligned[i]
        r1 = r1_w_aligned[i]
        s1 = s1_w_aligned[i]
        adx_val = adx[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # ADX trend strength filter
        strong_trend = adx_val > 25
        
        if position == 0:
            # Enter long if price breaks above R1, strong trend, and volume confirmation
            if price > r1 and strong_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short if price breaks below S1, strong trend, and volume confirmation
            elif price < s1 and strong_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price returns to pivot point or trend weakens
            if price < pp or adx_val < 20:  # Trend weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price returns to pivot point or trend weakens
            if price > pp or adx_val < 20:  # Trend weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals