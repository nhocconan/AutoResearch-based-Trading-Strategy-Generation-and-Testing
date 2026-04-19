#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In trending markets (ADX>25),
# we fade extreme readings (buy at %R < -80, sell at %R > -20) only when aligned with 12h trend.
# Volume confirmation ensures institutional participation. Designed to work in both bull/bear
# markets by combining mean-reversion in trends with trend filters.
# Target: 60-120 total trades over 4 years (15-30/year).
name = "6h_WilliamsR_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

def calculate_williams_r(high, low, close, period=14):
    """Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr.fillna(0).values

def calculate_adx(high, low, close, period=14):
    """ADX calculation using Wilder's smoothing"""
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
    valid = (plus_di[period:] + minus_di[period:]) > 0
    dx[period:] = np.where(valid, 100 * np.abs(plus_di[period:] - minus_di[period:]) / (plus_di[period:] + minus_di[period:]), 0)
    
    adx = np.zeros_like(close)
    if len(dx) >= 2*period+1:
        adx[2*period] = np.mean(dx[period:2*period+1])
        for i in range(2*period+1, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate EMA50 on 12h for trend direction
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate ADX on 12h for trend strength
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    
    # Calculate Williams %R on 6h
    wr = calculate_williams_r(high, low, close, 14)
    
    # Align 12h indicators to 6h
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(wr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_50_val = ema_50_12h_aligned[i]
        adx_val = adx_12h_aligned[i]
        wr_val = wr[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.3 * vol_ma
        
        # Trend filter: strong uptrend (price > EMA50 and ADX > 20) or strong downtrend
        strong_uptrend = price > ema_50_val and adx_val > 20
        strong_downtrend = price < ema_50_val and adx_val > 20
        
        if position == 0:
            # Enter long: Williams %R oversold (< -80) in uptrend with volume
            if wr_val < -80 and strong_uptrend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R overbought (> -20) in downtrend with volume
            elif wr_val > -20 and strong_downtrend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R overbought (> -20) or trend breaks
            if wr_val > -20 or not strong_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R oversold (< -80) or trend breaks
            if wr_val < -80 or not strong_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals