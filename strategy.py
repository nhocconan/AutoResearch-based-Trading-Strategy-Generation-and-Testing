#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Exponential Moving Average crossover with volume confirmation and ADX trend filter
# - Uses 1d EMA(34) for long-term trend direction
# - Uses 4h EMA(13) for entry timing in direction of 1d trend
# - Uses 4h volume spike to confirm institutional participation
# - Uses 4h ADX > 20 to filter for trending markets (lower threshold for more signals in weak trends)
# - Enters long when 1d EMA(34) rising AND 4h EMA(13) crosses above 4h EMA(34) with volume and ADX
# - Enters short when 1d EMA(34) falling AND 4h EMA(13) crosses below 4h EMA(34) with volume and ADX
# - Exits when 4h EMA(13) crosses back in opposite direction or ADX weakens (<20)
# - Designed to capture trend moves with lower whipsaw in both bull and bear markets
# - Target: 100-200 total trades over 4 years (25-50/year) with 0.30 position sizing

name = "4h_1dEMA34_4hEMA13_34_Volume_ADX"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend direction
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_rising = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_1d_falling = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_1d_rising[1:] = ema_34_1d[1:] > ema_34_1d[:-1]
    ema_34_1d_falling[1:] = ema_34_1d[1:] < ema_34_1d[:-1]
    
    # Align 1d EMA34 and its direction to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_34_1d_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_rising.astype(float))
    ema_34_1d_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_falling.astype(float))
    
    # 4h EMA(13) and EMA(34) for entry signals
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume filter (4h timeframe)
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_spike = volume > (1.5 * vol_ma_10)  # Volume confirmation
    
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
    adx_filter = adx_values > 20  # Trend filter (lowered to 20 for more signals in weak trends)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_34_1d_rising_aligned[i]) or 
            np.isnan(ema_34_1d_falling_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(ema_34[i]) or np.isnan(volume_spike[i]) or np.isnan(adx_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1d EMA34 rising AND 4h EMA13 crosses above EMA34 with volume and ADX
            if (ema_34_1d_rising_aligned[i] and 
                ema_13[i] > ema_34[i] and ema_13[i-1] <= ema_34[i-1] and
                volume_spike[i] and adx_filter[i]):
                signals[i] = 0.30
                position = 1
            # Short: 1d EMA34 falling AND 4h EMA13 crosses below EMA34 with volume and ADX
            elif (ema_34_1d_falling_aligned[i] and 
                  ema_13[i] < ema_34[i] and ema_13[i-1] >= ema_34[i-1] and
                  volume_spike[i] and adx_filter[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: EMA13 crosses back below EMA34 OR ADX weakens
            if ema_13[i] < ema_34[i] or not adx_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: EMA13 crosses back above EMA34 OR ADX weakens
            if ema_13[i] > ema_34[i] or not adx_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals