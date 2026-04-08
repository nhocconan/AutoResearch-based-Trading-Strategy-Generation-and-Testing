# 12h_1d_camarilla_pivot_volume_v1
# Hypothesis: Camarilla pivot levels from 1-day timeframe with volume confirmation and ADX trend filter captures institutional order flow reversals. Works in bull/bear markets by fading extremes during ranging conditions while avoiding countertrend trades during strong trends. 12h timeframe reduces overtrading; pivot levels provide high-probability reversal zones.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Using previous day's OHLC (standard Camarilla calculation)
    prev_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_low = df_1d['low'].shift(1).values    # Previous day's low
    prev_close = df_1d['close'].shift(1).values # Previous day's close
    
    # Camarilla levels
    # Resistance levels: R1 = C + (H-L)*1.1/12, R2 = C + (H-L)*1.1/6, R3 = C + (H-L)*1.1/4, R4 = C + (H-L)*1.1/2
    # Support levels: S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    hl_range = prev_high - prev_low
    r1 = prev_close + hl_range * 1.1 / 12
    r2 = prev_close + hl_range * 1.1 / 6
    r3 = prev_close + hl_range * 1.1 / 4
    r4 = prev_close + hl_range * 1.1 / 2
    s1 = prev_close - hl_range * 1.1 / 12
    s2 = prev_close - hl_range * 1.1 / 6
    s3 = prev_close - hl_range * 1.1 / 4
    s4 = prev_close - hl_range * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    # ADX trend filter: avoid trading against strong trends
    # Calculate ADX(14) on 12h data
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    period = 14
    if n >= period:
        # Initial ATR
        atr[period] = np.mean(tr[1:period+1])
        # Initial +DM and -DM
        plus_dm_sum = np.sum(plus_dm[1:period+1])
        minus_dm_sum = np.sum(minus_dm[1:period+1])
        plus_di[period] = 100 * plus_dm_sum / atr[period] if atr[period] != 0 else 0
        minus_di[period] = 100 * minus_dm_sum / atr[period] if atr[period] != 0 else 0
        
        # Smooth subsequent values
        for i in range(period+1, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_di[i] = (plus_di[i-1] * (period-1) + plus_dm[i]) / period
            minus_di[i] = (minus_di[i-1] * (period-1) + minus_dm[i]) / period
        
        # Calculate DX and ADX
        for i in range(period, n):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
            else:
                dx[i] = 0
        
        # Smooth DX to get ADX
        if n >= 2*period:
            adx[2*period] = np.mean(dx[period:2*period+1])
            for i in range(2*period+1, n):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # ADX threshold: only trade when ADX < 25 (ranging market)
    ranging_filter = adx < 25
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(adx[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S3 (strong support broken) or reaches R3 (profit target)
            if close[i] < s3_aligned[i] or close[i] > r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 (strong resistance broken) or reaches S3 (profit target)
            if close[i] > r3_aligned[i] or close[i] < s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Only trade in ranging market (ADX < 25)
            if ranging_filter[i]:
                # Long entry: price touches S3 with volume confirmation (bounce from strong support)
                if abs(close[i] - s3_aligned[i]) < (hl_range[i] * 0.001 if not np.isnan(hl_range[i]) else 0.01) and vol_confirm[i]:
                    # Additional check: price should be above S4 (not broken below)
                    if close[i] > s4_aligned[i]:
                        position = 1
                        signals[i] = 0.25
                # Short entry: price touches R3 with volume confirmation (rejection from strong resistance)
                elif abs(close[i] - r3_aligned[i]) < (hl_range[i] * 0.001 if not np.isnan(hl_range[i]) else 0.01) and vol_confirm[i]:
                    # Additional check: price should be below R4 (not broken above)
                    if close[i] < r4_aligned[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals