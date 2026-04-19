#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + volume spike + ADX trend filter
# Williams %R identifies overbought/oversold conditions for mean reversion in trends
# Volume spike confirms momentum, ADX > 25 ensures trending market (avoids chop)
# Works in bull/bear by fading extremes in strong trends with volume confirmation
# Target: 20-35 trades/year per symbol
name = "12h_WilliamsR_ADX25_Volume_Spike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
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
        wr = np.where((highest_high - lowest_low) != 0,
                      -100 * (highest_high - close) / (highest_high - lowest_low),
                      -50)
        return wr
    
    wr = williams_r(high, low, close, 14)
    
    # ADX calculation (14-period)
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
        valid = (plus_di + minus_di) != 0
        dx[period:] = np.where(valid[period:], 
                               100 * np.abs(plus_di[period:] - minus_di[period:]) / (plus_di[period:] + minus_di[period:]), 
                               0)
        
        adx = np.zeros_like(close)
        adx[2*period] = np.mean(dx[period:2*period+1])
        for i in range(2*period+1, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # WR(14) needs 27, ADX needs 28, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(wr[i]) or np.isnan(adx[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        wr_val = wr[i]
        adx_val = adx[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume spike confirmation
        volume_spike = vol > 2.0 * vol_ma
        
        # ADX trend strength filter
        strong_trend = adx_val > 25
        
        if position == 0:
            # Enter long when Williams %R oversold (< -80) in strong trend with volume spike
            if wr_val < -80 and strong_trend and volume_spike:
                signals[i] = 0.25
                position = 1
            # Enter short when Williams %R overbought (> -20) in strong trend with volume spike
            elif wr_val > -20 and strong_trend and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when Williams %R returns to neutral (> -50) or trend weakens
            if wr_val > -50 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when Williams %R returns to neutral (< -50) or trend weakens
            if wr_val < -50 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals