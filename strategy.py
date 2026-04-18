# 12h_Camarilla_R1_S1_Breakout_Volume_Regime_V1
# Hypothesis: Use 1d Camarilla pivot levels (R1/S1) for key support/resistance, with volume confirmation and 1d ADX trend filter.
# Long when price breaks above R1 with volume > 1.5x average and ADX > 25 (trending).
# Short when price breaks below S1 with volume > 1.5x average and ADX > 25.
# Exit when price returns to the Pivot point (PP) or ADX drops below 20 (range).
# Uses 12h timeframe for lower frequency to reduce fee churn. Designed to work in both bull and bear markets via long/short symmetry.
# Target: 15-30 trades/year to stay within optimal range and avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (using previous day's OHLC)
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pp_1d = np.full_like(close_1d, np.nan)
    r1_1d = np.full_like(close_1d, np.nan)
    s1_1d = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        # Use previous day's data to avoid look-ahead
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        pp_1d[i] = (ph + pl + pc) / 3.0
        r1_1d[i] = pc + (ph - pl) * 1.1 / 12.0
        s1_1d[i] = pc - (ph - pl) * 1.1 / 12.0
    
    # Calculate 1d ADX(14) for trend strength
    # +DM = max(0, high[i] - high[i-1]) if > max(0, low[i-1] - low[i])
    # -DM = max(0, low[i-1] - low[i]) if > max(0, high[i] - high[i-1])
    # TR = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    # +DI = 100 * smoothed(+DM) / smoothed(TR)
    # -DI = 100 * smoothed(-DM) / smoothed(TR)
    # DX = 100 * abs(+DI - -DI) / (+DI + -DI)
    # ADX = smoothed(DX)
    adx_period = 14
    tr = np.full_like(high_1d, np.nan)
    plus_dm = np.full_like(high_1d, np.nan)
    minus_dm = np.full_like(high_1d, np.nan)
    
    for i in range(1, len(high_1d)):
        hl = high_1d[i] - low_1d[i]
        hc = abs(high_1d[i] - close_1d[i-1])
        lc = abs(low_1d[i] - close_1d[i-1])
        tr[i] = max(hl, hc, lc)
        
        up = high_1d[i] - high_1d[i-1]
        down = low_1d[i-1] - low_1d[i]
        plus_dm[i] = up if up > down and up > 0 else 0
        minus_dm[i] = down if down > up and down > 0 else 0
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[1:period])  # Skip index 0 as it's undefined
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period - 1) + data[i]) / period
        return result
    
    tr_smooth = wilders_smoothing(tr, adx_period)
    plus_dm_smooth = wilders_smoothing(plus_dm, adx_period)
    minus_dm_smooth = wilders_smoothing(minus_dm, adx_period)
    
    # Avoid division by zero
    plus_di = np.full_like(tr_smooth, np.nan)
    minus_di = np.full_like(tr_smooth, np.nan)
    dx = np.full_like(tr_smooth, np.nan)
    
    for i in range(len(tr_smooth)):
        if not np.isnan(tr_smooth[i]) and tr_smooth[i] != 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # Smooth DX to get ADX
    adx = wilders_smoothing(dx, adx_period)
    
    # Align 1d indicators to 12h timeframe
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(vol_period, 1) + 1  # Start after volume MA and pivot calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and trend (ADX > 25)
            if close[i] > r1_1d_aligned[i] and vol_confirm and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and trend (ADX > 25)
            elif close[i] < s1_1d_aligned[i] and vol_confirm and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to PP or trend weakens (ADX < 20)
            if close[i] <= pp_1d_aligned[i] or adx_aligned[i] < 20:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to PP or trend weakens (ADX < 20)
            if close[i] >= pp_1d_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_Regime_V1"
timeframe = "12h"
leverage = 1.0