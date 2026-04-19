#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 12h trend filter (ADX > 25) and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In strong trends (ADX > 25),
# we fade extreme readings only when aligned with the trend direction, avoiding counter-trend trades.
# Volume confirmation ensures momentum behind the move. Designed for range-bound markets with
# occasional trends, works in both bull and bear by filtering weak trends.
# Target: 20-40 trades/year per symbol.
name = "6h_WilliamsR_12hADX25_Trend_Volume"
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
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # ADX calculation (14-period) on 12h
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
        valid = (plus_di[period:] + minus_di[period:]) != 0
        dx[period:] = np.where(valid, 100 * np.abs(plus_di[period:] - minus_di[period:]) / (plus_di[period:] + minus_di[period:]), 0)
        
        adx = np.zeros_like(close)
        if len(dx) >= 2*period+1:
            adx[2*period] = np.mean(dx[period:2*period+1])
            for i in range(2*period+1, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    
    # Williams %R (14-period) on 6h
    def williams_r(high, low, close, period=14):
        highest_high = np.zeros_like(high)
        lowest_low = np.zeros_like(low)
        for i in range(len(high)):
            if i < period:
                highest_high[i] = np.nan
                lowest_low[i] = np.nan
            else:
                highest_high[i] = np.max(high[i-period+1:i+1])
                lowest_low[i] = np.min(low[i-period+1:i+1])
        wr = np.zeros_like(close)
        denom = highest_high - lowest_low
        wr = np.where(denom != 0, -100 * (highest_high - close) / denom, -50)
        return wr
    
    wr = williams_r(high, low, close, 14)
    
    # Align 12h ADX to 6h
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure Williams %R, ADX, and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(wr[i]) or np.isnan(adx_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        wr_val = wr[i]
        adx_val = adx_12h_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.3 * vol_ma
        
        # Strong trend filter
        strong_trend = adx_val > 25
        
        if position == 0:
            # Enter long: Williams %R oversold (< -80) AND strong trend AND volume confirmation
            if wr_val < -80 and strong_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R overbought (> -20) AND strong trend AND volume confirmation
            elif wr_val > -20 and strong_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when Williams %R returns to neutral (> -50) or trend weakens
            if wr_val > -50 or adx_val < 20:  # Trend weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when Williams %R returns to neutral (< -50) or trend weakens
            if wr_val < -50 or adx_val < 20:  # Trend weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals