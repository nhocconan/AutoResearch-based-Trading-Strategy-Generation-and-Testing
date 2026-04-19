#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In strong trends (ADX > 25),
# we fade extreme readings only when price is near 1d EMA50 (dynamic support/resistance).
# Volume confirms the mean reversion bounce. Works in both bull/bear markets by
# adapting to trend strength and avoiding counter-trend trades in weak trends.
# Target: 15-35 trades/year per symbol (60-140 total over 4 years).
name = "6h_WilliamsR_MeanReversion_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def williams_r(high, low, close, period=14):
    """Williams %R indicator"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ADX calculation (14-period) for trend strength filter
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
    
    # Williams %R (14-period)
    wr = williams_r(high, low, close, 14)
    
    # Align 1d EMA50 to 6h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 28)  # Ensure EMA50 and ADX are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(adx[i]) or np.isnan(wr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_50_val = ema_50_aligned[i]
        adx_val = adx[i]
        wr_val = wr[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.3 * vol_ma
        
        # ADX trend strength filter (only trade in strong trends)
        strong_trend = adx_val > 25
        
        # Williams %R levels
        oversold = wr_val <= -80
        overbought = wr_val >= -20
        
        # Price near 1d EMA50 (within 1.5%)
        price_near_ema = abs(price - ema_50_val) / ema_50_val < 0.015
        
        if position == 0:
            # Enter long: oversold + near EMA50 + strong trend + volume
            if oversold and price_near_ema and strong_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: overbought + near EMA50 + strong trend + volume
            elif overbought and price_near_ema and strong_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to neutral or trend weakens
            if wr_val >= -50 or adx_val < 20:  # Return from oversold or weakening trend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to neutral or trend weakens
            if wr_val <= -50 or adx_val < 20:  # Return from overbought or weakening trend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals