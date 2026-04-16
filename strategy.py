#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily data for pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Previous Day Values for Pivot Calculation ===
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    # First day uses current day values (no look-ahead)
    prev_close_1d[0] = close_1d[0]
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    # === Daily Pivot Points ===
    pivot_point = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    prev_range = prev_high_1d - prev_low_1d
    
    # === Resistance/Support Levels ===
    r1 = pivot_point + prev_range * 0.382
    s1 = pivot_point - prev_range * 0.382
    r2 = pivot_point + prev_range * 0.618
    s2 = pivot_point - prev_range * 0.618
    
    # Align levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # === Volume Confirmation (1h) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    # === ADX Trend Filter (Daily) ===
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]
    
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr14 = wilders_smooth(tr, period)
    dm_plus14 = wilders_smooth(dm_plus, period)
    dm_minus14 = wilders_smooth(dm_minus, period)
    
    tr14_safe = np.where(tr14 == 0, 1, tr14)
    di_plus = 100 * dm_plus14 / tr14_safe
    di_minus = 100 * dm_minus14 / tr14_safe
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilders_smooth(dx, period)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === Session Filter: 08-20 UTC ===
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    warmup = 50
    
    position = 0
    entry_price = 0.0
    
    for i in range(warmup, n):
        if not in_session[i]:
            if position == 1:
                signals[i] = 0.0
                position = 0
            elif position == -1:
                signals[i] = 0.0
                position = 0
            continue
        
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.0
                position = 0
            elif position == -1:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        adx_val = adx_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Exit logic
        if position == 1:
            if price < s1_val or price > r2_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            if price > r1_val or price < s2_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Entry logic
            if adx_val > 25 and vol_ratio_val > 2.0:
                if price > r1_val:
                    signals[i] = 0.20
                    position = 1
                    entry_price = price
                elif price < s1_val:
                    signals[i] = -0.20
                    position = -1
                    entry_price = price
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_FibPivot_Volume_ADX_Session"
timeframe = "1h"
leverage = 1.0