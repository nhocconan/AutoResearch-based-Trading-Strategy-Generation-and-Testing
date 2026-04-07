#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot Reversal with 12h Trend and Volume Filter
# Hypothesis: Price reverts from extreme Camarilla levels (R3/S3) in ranging markets,
# but breaks out from R4/S4 in trending markets. Uses 12h trend to filter direction
# and volume to confirm institutional participation. Works in both bull and bear:
# - In ranging markets (ADX < 25): fade R3/S3 entries
# - In trending markets (ADX >= 25): breakout R4/S4 entries
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_camarilla_pivot_reversal_trend_volume_v1"
timeframe = "6h"
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
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h OHLC
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar
    # R4 = C + (H-L) * 1.1/2
    # R3 = C + (H-L) * 1.1/4
    # S3 = C - (H-L) * 1.1/4
    # S4 = C - (H-L) * 1.1/2
    diff_12h = high_12h - low_12h
    r4_12h = close_12h + diff_12h * 1.1 / 2
    r3_12h = close_12h + diff_12h * 1.1 / 4
    s3_12h = close_12h - diff_12h * 1.1 / 4
    s4_12h = close_12h - diff_12h * 1.1 / 2
    
    # Shift by 1 to use previous 12h bar's levels (avoid look-ahead)
    r4_12h_prev = np.roll(r4_12h, 1)
    r3_12h_prev = np.roll(r3_12h, 1)
    s3_12h_prev = np.roll(s3_12h, 1)
    s4_12h_prev = np.roll(s4_12h, 1)
    # Handle first bar
    if len(r4_12h_prev) > 1:
        r4_12h_prev[0] = r4_12h_prev[1]
        r3_12h_prev[0] = r3_12h_prev[1]
        s3_12h_prev[0] = s3_12h_prev[1]
        s4_12h_prev[0] = s4_12h_prev[1]
    else:
        r4_12h_prev[0] = 0
        r3_12h_prev[0] = 0
        s3_12h_prev[0] = 0
        s4_12h_prev[0] = 0
    
    # Align Camarilla levels to 6h timeframe
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h_prev)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h_prev)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h_prev)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h_prev)
    
    # 12h trend filter: EMA25 crossover
    close_12h_series = pd.Series(close_12h)
    ema_25_12h = close_12h_series.ewm(span=25, min_periods=25, adjust=False).mean().values
    ema_25_12h_prev = np.roll(ema_25_12h, 1)
    if len(ema_25_12h_prev) > 1:
        ema_25_12h_prev[0] = ema_25_12h_prev[1]
    else:
        ema_25_12h_prev[0] = 0
    ema_25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_25_12h_prev)
    
    # ADX for regime detection (12h)
    # Calculate +DM, -DM, TR
    plus_dm = np.zeros(len(high_12h))
    minus_dm = np.zeros(len(high_12h))
    tr = np.zeros(len(high_12h))
    for i in range(1, len(high_12h)):
        plus_dm[i] = max(0, high_12h[i] - high_12h[i-1])
        minus_dm[i] = max(0, low_12h[i-1] - low_12h[i])
        tr[i] = max(high_12h[i] - low_12h[i], 
                    abs(high_12h[i] - close_12h[i-1]), 
                    abs(low_12h[i] - close_12h[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[1:period])
        # Wilder smoothing: today = (yesterday * (period-1) + today) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Calculate smoothed +DM, -DM, TR
    atr_period = 14
    plus_dm_smooth = wilders_smooth(plus_dm, atr_period)
    minus_dm_smooth = wilders_smooth(minus_dm, atr_period)
    tr_smooth = wilders_smooth(tr, atr_period)
    
    # Calculate +DI and -DI
    plus_di = np.zeros_like(tr_smooth)
    minus_di = np.zeros_like(tr_smooth)
    for i in range(len(tr_smooth)):
        if tr_smooth[i] != 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
        else:
            plus_di[i] = 0
            minus_di[i] = 0
    
    # Calculate DX and ADX
    dx = np.zeros_like(tr_smooth)
    for i in range(len(tr_smooth)):
        if (plus_di[i] + minus_di[i]) != 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        else:
            dx[i] = 0
    adx = wilders_smooth(dx, atr_period)
    
    # Shift ADX by 1 to use previous value
    adx_prev = np.roll(adx, 1)
    if len(adx_prev) > 1:
        adx_prev[0] = adx_prev[1]
    else:
        adx_prev[0] = 0
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_prev)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(25, n):
        # Skip if required data not available
        if (np.isnan(r4_12h_aligned[i]) or np.isnan(r3_12h_aligned[i]) or 
            np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(ema_25_12h_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions
            exit_condition = False
            # Mean reversion exit: price crosses below R3 in ranging market
            if adx_aligned[i] < 25 and close[i] < r3_12h_aligned[i]:
                exit_condition = True
            # Trend following exit: price crosses below S4 in trending market
            elif adx_aligned[i] >= 25 and close[i] < s4_12h_aligned[i]:
                exit_condition = True
            # Trend filter exit: price below 12h EMA25
            elif close[i] < ema_25_12h_aligned[i]:
                exit_condition = True
            # Volume filter exit
            elif not vol_filter[i]:
                exit_condition = True
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_condition = False
            # Mean reversion exit: price crosses above S3 in ranging market
            if adx_aligned[i] < 25 and close[i] > s3_12h_aligned[i]:
                exit_condition = True
            # Trend following exit: price crosses above R4 in trending market
            elif adx_aligned[i] >= 25 and close[i] > r4_12h_aligned[i]:
                exit_condition = True
            # Trend filter exit: price above 12h EMA25
            elif close[i] > ema_25_12h_aligned[i]:
                exit_condition = True
            # Volume filter exit
            elif not vol_filter[i]:
                exit_condition = True
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
                
        else:  # Flat, look for entry
            # Determine market regime
            is_ranging = adx_aligned[i] < 25
            is_trending = adx_aligned[i] >= 25
            
            # Mean reversion entries (ranging market): fade extremes at R3/S3
            if is_ranging:
                # Long: price rejects S3 with volume confirmation
                if (low[i] <= s3_12h_aligned[i] * 1.001 and  # Allow small tolerance
                    close[i] > s3_12h_aligned[i] and 
                    vol_filter[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price rejects R3 with volume confirmation
                elif (high[i] >= r3_12h_aligned[i] * 0.999 and  # Allow small tolerance
                      close[i] < r3_12h_aligned[i] and 
                      vol_filter[i]):
                    position = -1
                    signals[i] = -0.25
            
            # Trend following entries (trending market): breakout R4/S4
            elif is_trending:
                # Long: price breaks above R4 with volume and trend alignment
                if (high[i] > r4_12h_aligned[i] and 
                    close[i] > ema_25_12h_aligned[i] and 
                    vol_filter[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below S4 with volume and trend alignment
                elif (low[i] < s4_12h_aligned[i] and 
                      close[i] < ema_25_12h_aligned[i] and 
                      vol_filter[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals