#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band squeeze breakout with 1w volume confirmation and ADX trend filter
# Designed for low trade frequency (target 10-25/year) with clear breakout logic
# Works in both bull (breakout above upper band in uptrend) and bear (breakdown below lower band in downtrend) markets
# Uses Bollinger Bands from daily, volume spike to confirm breakout, and weekly ADX for trend strength

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Bollinger Bands and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Bollinger Bands (20, 2) on 1d - using previous day's data to avoid look-ahead
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    
    # Calculate Bollinger Band width for squeeze detection
    bb_width = (upper_band - lower_band) / sma_20
    
    # Calculate ADX (14) on 1w for trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] > minus_dm[i]:
                minus_dm[i] = 0
            elif minus_dm[i] > plus_dm[i]:
                plus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (equivalent to alpha=1/period)
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        atr[period] = np.nansum(tr[1:period+1]) if period < len(tr) else 0
        plus_dm_sum = np.nansum(plus_dm[1:period+1]) if period < len(plus_dm) else 0
        minus_dm_sum = np.nansum(minus_dm[1:period+1]) if period < len(minus_dm) else 0
        
        if atr[period] != 0:
            plus_di[period] = plus_dm_sum * 100 / atr[period]
            minus_di[period] = minus_dm_sum * 100 / atr[period]
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_di[i] = (plus_di[i-1] * (period-1) + plus_dm[i]) * 100 / atr[i] if atr[i] != 0 else 0
            minus_di[i] = (minus_di[i-1] * (period-1) + minus_dm[i]) * 100 / atr[i] if atr[i] != 0 else 0
        
        dx = np.zeros_like(high)
        for i in range(period, len(high)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = abs(plus_di[i] - minus_di[i]) * 100 / (plus_di[i] + minus_di[i])
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.nansum(dx[period:2*period]) if (2*period-1) < len(dx) else 0
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period if adx[i-1] != 0 else dx[i]
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Align all indicators to main timeframe (1d prices)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):  # Start after enough data for indicators
        # Skip if any required data is NaN
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(bb_width_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            continue
        
        # Long entry: price breaks above upper band during low volatility (squeeze) + strong trend
        if (close[i] > upper_band_aligned[i] and 
            bb_width_aligned[i] < 0.05 and  # Bollinger Band squeeze (<5% width)
            adx_1w_aligned[i] > 25 and      # Strong trend
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and  # Volume spike
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below lower band during low volatility (squeeze) + strong trend
        elif (close[i] < lower_band_aligned[i] and 
              bb_width_aligned[i] < 0.05 and  # Bollinger Band squeeze (<5% width)
              adx_1w_aligned[i] > 25 and      # Strong trend
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and  # Volume spike
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or volatility expansion (end of squeeze)
        elif position == 1 and bb_width_aligned[i] > 0.10:  # Volatility expansion
            position = 0
            signals[i] = 0.0
        elif position == -1 and bb_width_aligned[i] > 0.10:  # Volatility expansion
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_Bollinger_Squeeze_1wADX_Volume_Breakout"
timeframe = "1d"
leverage = 1.0