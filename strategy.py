#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with weekly trend filter and volume confirmation
# Uses 1-week ADX for trend strength to avoid choppy markets
# Breakout provides clear entry/exit with low trade frequency (target: 15-25/year)
# Works in bull/bear via trend filter - only trades in direction of weekly trend

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ADX(14) for trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] > minus_dm[i]:
                minus_dm[i] = 0
            elif minus_dm[i] > plus_dm[i]:
                plus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
                
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_sum = np.sum(plus_dm[1:period+1])
        minus_dm_sum = np.sum(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - (plus_dm_sum / period) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm_sum / period) + minus_dm[i]
            plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
        
        dx = np.zeros_like(high)
        adx = np.zeros_like(high)
        dx_sum = 0
        
        for i in range(2*period, len(high)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
            else:
                dx[i] = 0
            
            if i == 2*period:
                dx_sum = np.sum(dx[period+1:2*period+1])
                adx[i] = dx_sum / period
            else:
                dx_sum = dx_sum - (dx_sum / period) + dx[i]
                adx[i] = dx_sum / period
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Calculate 12-hour Donchian channels (20-period)
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    
    for i in range(20, len(high)):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation (20-period average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    vol_spike = volume > 1.5 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align weekly ADX to 12h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        # Skip if data not ready or outside session
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(adx_1w_aligned[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade when weekly trend is strong (ADX > 25)
        if adx_1w_aligned[i] > 25:
            if position == 0:
                # Long: Break above Donchian high + volume spike
                if close[i] > donchian_high[i] and vol_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Break below Donchian low + volume spike
                elif close[i] < donchian_low[i] and vol_spike[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                # Exit: Return to opposite Donchian level
                if position == 1:
                    if close[i] < donchian_low[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                else:  # position == -1
                    if close[i] > donchian_high[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
        else:
            # Weak trend - exit any position
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Trend_ADX_Volume_Session"
timeframe = "12h"
leverage = 1.0