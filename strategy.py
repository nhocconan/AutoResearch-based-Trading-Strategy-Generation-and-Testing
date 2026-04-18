# 1d_Pivot_R1_S1_Breakout_Volume_RangeFilter_Strict
# Strategy: Camarilla pivot breakout with volume confirmation and range filter
# Long: Price breaks above R1 with volume > 2x average, in trending market (ADX > 25)
# Short: Price breaks below S1 with volume > 2x average, in trending market (ADX > 25)
# Uses 1d timeframe with 1h trend filter via ADX
# Expected: 5-15 trades/year per symbol, works in bull/bear via trend filter

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = max(high_diff, 0) if high_diff > low_diff else 0
        minus_dm[i] = max(low_diff, 0) if low_diff > high_diff else 0
        
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    atr = np.zeros_like(tr)
    atr[period] = np.mean(tr[1:period+1])
    for i in range(period+1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_di = np.zeros_like(high)
    minus_di = np.zeros_like(high)
    if period < len(high):
        plus_sm = np.zeros_like(high)
        minus_sm = np.zeros_like(high)
        plus_sm[period] = np.sum(plus_dm[1:period+1])
        minus_sm[period] = np.sum(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            plus_sm[i] = (plus_sm[i-1] * (period-1) + plus_dm[i]) / period
            minus_sm[i] = (minus_sm[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_sm / atr
        minus_di = 100 * minus_sm / atr
    
    dx = np.zeros_like(high)
    mask = (plus_di + minus_di) != 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask])
    
    adx = np.zeros_like(high)
    if 2*period < len(dx):
        adx[2*period] = np.mean(dx[period:2*period+1])
        for i in range(2*period+1, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's values (shift by 1)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First day has no previous data
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Calculate Camarilla levels
    camarilla_range = prev_high - prev_low
    R1 = prev_close + camarilla_range * 1.1 / 12
    S1 = prev_close - camarilla_range * 1.1 / 12
    
    # Align to 1d timeframe (no shift needed as we use previous day's data)
    R1_aligned = R1  # Already aligned to daily bars
    S1_aligned = S1
    
    # Get 1h data for ADX trend filter
    df_1h = get_htf_data(prices, '1h')
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate ADX on 1h data
    adx_1h = calculate_adx(high_1h, low_1h, close_1h, 14)
    
    # Align ADX to 1d timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1h, adx_1h)
    
    # Volume confirmation - 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough for ADX and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_1d_aligned[i] > 25
        
        # Volume confirmation: volume > 2x average
        vol_confirm = volume[i] > 2.0 * vol_ma_20[i]
        
        # Breakout conditions
        breakout_long = close[i] > R1_aligned[i]
        breakout_short = close[i] < S1_aligned[i]
        
        if position == 0:
            # Long: breakout above R1 + volume + trending
            if breakout_long and vol_confirm and trending:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + volume + trending
            elif breakout_short and vol_confirm and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below R1 or trend weakens
            if close[i] < R1_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above S1 or trend weakens
            if close[i] > S1_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Pivot_R1_S1_Breakout_Volume_RangeFilter_Strict"
timeframe = "1d"
leverage = 1.0