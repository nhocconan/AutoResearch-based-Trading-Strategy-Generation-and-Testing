#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h ADX for trend strength and 4h Supertrend for direction,
# filtered by 1d EMA200 and volume confirmation. Uses 1h only for entry timing precision.
# Designed to work in both bull and bear markets by filtering weak trends and choppy markets.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years).
name = "1h_ADX25_Supertrend_EMA200_Volume_Filter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for ADX and Supertrend
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ADX (14-period) on 4h
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
    
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    
    # Calculate Supertrend (ATR=10, multiplier=3) on 4h
    def calculate_supertrend(high, low, close, atr_period=10, multiplier=3):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(close)
        atr[atr_period] = np.mean(tr[1:atr_period+1])
        for i in range(atr_period+1, len(tr)):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
        
        hl2 = (high + low) / 2
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr
        
        upper_band_final = np.zeros_like(close)
        lower_band_final = np.zeros_like(close)
        supertrend = np.zeros_like(close)
        trend = np.ones_like(close)  # 1 for up, -1 for down
        
        upper_band_final[0] = upper_band[0]
        lower_band_final[0] = lower_band[0]
        supertrend[0] = lower_band[0]
        trend[0] = 1
        
        for i in range(1, len(close)):
            if close[i] > upper_band_final[i-1]:
                trend[i] = 1
            elif close[i] < lower_band_final[i-1]:
                trend[i] = -1
            else:
                trend[i] = trend[i-1]
            
            if trend[i] == 1:
                upper_band_final[i] = max(upper_band[i], upper_band_final[i-1])
                lower_band_final[i] = lower_band[i]
                supertrend[i] = upper_band_final[i]
            else:
                upper_band_final[i] = upper_band[i]
                lower_band_final[i] = min(lower_band[i], lower_band_final[i-1])
                supertrend[i] = lower_band_final[i]
        
        return supertrend, trend
    
    supertrend_4h, trend_4h = calculate_supertrend(high_4h, low_4h, close_4h, 10, 3)
    
    # Get 1d data for EMA200
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 4h indicators to 1h
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    supertrend_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend_4h)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 28, 20)  # Ensure EMA200, ADX, and Supertrend are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(supertrend_4h_aligned[i]) or 
            np.isnan(trend_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx_4h_aligned[i]
        supertrend_val = supertrend_4h_aligned[i]
        trend_val = trend_4h_aligned[i]
        ema_200_val = ema_200_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        hour = hours[i]
        
        # Session filter: only trade between 8-20 UTC
        in_session = (8 <= hour <= 20)
        
        # Volume confirmation
        volume_confirmed = vol > 1.5 * vol_ma
        
        # ADX trend strength filter
        strong_trend = adx_val > 25
        
        if position == 0:
            # Enter long if Supertrend is up, price above Supertrend, strong trend, EMA200 filter, volume, and session
            if (trend_val == 1 and price > supertrend_val and strong_trend and 
                price > ema_200_val and volume_confirmed and in_session):
                signals[i] = 0.20
                position = 1
            # Enter short if Supertrend is down, price below Supertrend, strong trend, EMA200 filter, volume, and session
            elif (trend_val == -1 and price < supertrend_val and strong_trend and 
                  price < ema_200_val and volume_confirmed and in_session):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long when Supertrend flips down or trend weakens
            if trend_val == -1 or adx_val < 20:  # Trend weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short when Supertrend flips up or trend weakens
            if trend_val == 1 or adx_val < 20:  # Trend weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals