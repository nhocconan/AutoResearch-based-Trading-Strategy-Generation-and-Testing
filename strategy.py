#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation and ADX trend filter
# - Long when price breaks above Camarilla R4 AND 1d volume > 1.5x 20-period average AND 1d ADX > 25
# - Short when price breaks below Camarilla S4 AND 1d volume > 1.5x 20-period average AND 1d ADX > 25
# - Exit when price returns to Camarilla PP (pivot point)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Camarilla levels provide intraday support/resistance; breakouts indicate strong momentum
# - Daily volume confirms institutional participation; ADX ensures trending environment
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1d_camarilla_volume_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 6h ATR (14-period) for stoploss
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr = np.zeros_like(tr)
    atr[13] = np.mean(tr[1:15])  # First ATR value
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Pre-compute 1d typical price for Camarilla calculation
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    tp_high_1d = df_1d['high'].values
    tp_low_1d = df_1d['low'].values
    tp_close_1d = typical_price_1d.values
    
    # Pre-compute 1d Camarilla levels (based on previous day)
    camarilla_pp = np.full_like(tp_close_1d, np.nan, dtype=float)
    camarilla_r4 = np.full_like(tp_close_1d, np.nan, dtype=float)
    camarilla_s4 = np.full_like(tp_close_1d, np.nan, dtype=float)
    
    for i in range(1, len(tp_close_1d)):
        # Camarilla calculations based on previous day's OHLC
        high_prev = tp_high_1d[i-1]
        low_prev = tp_low_1d[i-1]
        close_prev = tp_close_1d[i-1]
        
        camarilla_pp[i] = (high_prev + low_prev + close_prev) / 3.0
        range_prev = high_prev - low_prev
        camarilla_r4[i] = camarilla_pp[i] + range_prev * 1.1 / 2.0
        camarilla_s4[i] = camarilla_pp[i] - range_prev * 1.1 / 2.0
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    def rolling_mean(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result
    
    vol_ma_1d = rolling_mean(volume_1d, 20)
    
    # Pre-compute 1d ADX (14-period)
    def calculate_adx(high_arr, low_arr, close_arr, period=14):
        # Calculate +DM and -DM
        plus_dm = np.zeros_like(high_arr)
        minus_dm = np.zeros_like(high_arr)
        
        for i in range(1, len(high_arr)):
            high_diff = high_arr[i] - high_arr[i-1]
            low_diff = low_arr[i-1] - low_arr[i]
            
            if high_diff > low_diff and high_diff > 0:
                plus_dm[i] = high_diff
            else:
                plus_dm[i] = 0
                
            if low_diff > high_diff and low_diff > 0:
                minus_dm[i] = low_diff
            else:
                minus_dm[i] = 0
        
        # Calculate TR
        tr_arr = np.zeros_like(high_arr)
        tr_arr[0] = high_arr[0] - low_arr[0]
        for i in range(1, len(high_arr)):
            tr_arr[i] = true_range(high_arr[i], low_arr[i], close_arr[i-1])
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        def wilders_smoothing(arr, period):
            result = np.full_like(arr, np.nan, dtype=float)
            if len(arr) >= period:
                result[period-1] = np.mean(arr[:period])
                for i in range(period, len(arr)):
                    result[i] = (result[i-1] * (period-1) + arr[i]) / period
            return result
        
        tr_smooth = wilders_smoothing(tr_arr, period)
        plus_dm_smooth = wilders_smoothing(plus_dm, period)
        minus_dm_smooth = wilders_smoothing(minus_dm, period)
        
        # Calculate DI+
        plus_di = np.full_like(high_arr, np.nan, dtype=float)
        minus_di = np.full_like(high_arr, np.nan, dtype=float)
        
        for i in range(len(tr_smooth)):
            if not np.isnan(tr_smooth[i]) and tr_smooth[i] != 0:
                plus_di[i] = (plus_dm_smooth[i] / tr_smooth[i]) * 100
                minus_di[i] = (minus_dm_smooth[i] / tr_smooth[i]) * 100
        
        # Calculate DX
        dx = np.full_like(high_arr, np.nan, dtype=float)
        for i in range(len(tr_smooth)):
            if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
                di_sum = plus_di[i] + minus_di[i]
                if di_sum != 0:
                    dx[i] = np.abs(plus_di[i] - minus_di[i]) / di_sum * 100
        
        # Calculate ADX
        adx = wilders_smoothing(dx, period)
        return adx
    
    adx_1d = calculate_adx(tp_high_1d, tp_low_1d, tp_close_1d, 14)
    
    # Align HTF indicators to 6h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Volume spike condition
            vol_ma_6h = rolling_mean(volume, 20)
            vol_spike = not np.isnan(vol_ma_6h[i]) and volume[i] > 1.5 * vol_ma_6h[i]
            
            # Long conditions: Camarilla R4 breakout AND volume spike AND 1d ADX > 25
            if (close[i] > camarilla_r4_aligned[i] and vol_spike and 
                adx_1d_aligned[i] > 25):
                position = 1
                signals[i] = 0.25
            # Short conditions: Camarilla S4 breakdown AND volume spike AND 1d ADX > 25
            elif (close[i] < camarilla_s4_aligned[i] and vol_spike and 
                  adx_1d_aligned[i] > 25):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to Camarilla PP
            exit_long = (position == 1 and close[i] <= camarilla_pp_aligned[i])
            exit_short = (position == -1 and close[i] >= camarilla_pp_aligned[i])
            
            # Optional: ATR-based stoploss
            stop_long = (position == 1 and close[i] <= high[i] - 2.5 * atr[i])
            stop_short = (position == -1 and close[i] >= low[i] + 2.5 * atr[i])
            
            if exit_long or exit_short or stop_long or stop_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

def rolling_mean(arr, window):
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        result[i] = np.mean(arr[i - window + 1:i + 1])
    return result

def true_range(h, l, c_prev):
    tr1 = h - l
    tr2 = np.abs(h - c_prev)
    tr3 = np.abs(l - c_prev)
    return np.maximum(tr1, np.maximum(tr2, tr3))