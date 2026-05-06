#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily Chaikin Money Flow (CMF) for institutional flow detection
# - Uses 1d CMF(20) to detect accumulation/distribution (>0.15 accumulation, <-0.15 distribution)
# - Uses 4h price action for entry timing with volume confirmation
# - Uses 4h ADX > 20 to avoid choppy markets
# - Enters long when 1d CMF > 0.15 and price closes above 4h VWAP with volume
# - Enters short when 1d CMF < -0.15 and price closes below 4h VWAP with volume
# - Exits when CMF returns to neutral zone (-0.05 to 0.05) or opposite signal
# - Designed to capture institutional flow shifts with confirmation
# - Target: 100-180 total trades over 4 years (25-45/year) with 0.25 position sizing

name = "4h_1dCMF_20_VWAP_Volume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Chaikin Money Flow
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Chaikin Money Flow (20-period)
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    mfm = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if high_1d[i] != low_1d[i]:
            mfm[i] = ((close_1d[i] - low_1d[i]) - (high_1d[i] - close_1d[i])) / (high_1d[i] - low_1d[i])
        else:
            mfm[i] = 0.0  # Avoid division by zero
    
    # Money Flow Volume = MFM * Volume
    mfv = mfm * volume_1d
    
    # CMF = 20-period sum of MFV / 20-period sum of Volume
    cmf = np.zeros_like(close_1d)
    for i in range(19, len(close_1d)):  # Start at index 19 for 20-period
        mfv_sum = np.sum(mfv[i-19:i+1])
        vol_sum = np.sum(volume_1d[i-19:i+1])
        if vol_sum != 0:
            cmf[i] = mfv_sum / vol_sum
        else:
            cmf[i] = 0.0
    
    # Align 1d CMF to 4h timeframe
    cmf_4h = align_htf_to_ltf(prices, df_1d, cmf)
    
    # VWAP calculation (4h timeframe)
    typical_price = (high + low + close) / 3
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.divide(vwap_numerator, vwap_denominator, out=np.zeros_like(vwap_numerator), where=vwap_denominator!=0)
    
    # Volume filter (4h timeframe)
    vol_ma_15 = pd.Series(volume).rolling(window=15, min_periods=15).mean().values
    volume_spike = volume > (1.5 * vol_ma_15)  # Volume confirmation
    
    # ADX filter (4h timeframe) - trend strength
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
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        atr[period-1] = np.mean(tr[1:period+1])
        plus_dm_sum = np.sum(plus_dm[1:period+1])
        minus_dm_sum = np.sum(minus_dm[1:period+1])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - (plus_dm[i-period+1] if i-period+1 >= 0 else 0) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm[i-period+1] if i-period+1 >= 0 else 0) + minus_dm[i]
            plus_di[i] = 100 * plus_dm_sum / (atr[i] * period) if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_sum / (atr[i] * period) if atr[i] != 0 else 0
        
        dx = np.zeros_like(high)
        adx = np.zeros_like(high)
        for i in range(2*period-1, len(high)):
            di_diff = abs(plus_di[i] - minus_di[i])
            di_sum = plus_di[i] + minus_di[i]
            dx[i] = 100 * di_diff / di_sum if di_sum != 0 else 0
        
        # Smooth DX to get ADX
        adx[2*period-1] = np.mean(dx[2*period-1:3*period]) if 3*period <= len(high) else 0
        for i in range(3*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_values = calculate_adx(high, low, close, 14)
    adx_filter = adx_values > 20  # Moderate trend filter to avoid chop
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(cmf_4h[i]) or np.isnan(vwap[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(adx_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: CMF accumulation + price above VWAP + volume + trend
            if cmf_4h[i] > 0.15 and close[i] > vwap[i] and volume_spike[i] and adx_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: CMF distribution + price below VWAP + volume + trend
            elif cmf_4h[i] < -0.15 and close[i] < vwap[i] and volume_spike[i] and adx_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: CMF returns to neutral or turns negative
            if cmf_4h[i] < 0.05 or cmf_4h[i] < -0.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: CMF returns to neutral or turns positive
            if cmf_4h[i] > -0.05 or cmf_4h[i] > 0.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals