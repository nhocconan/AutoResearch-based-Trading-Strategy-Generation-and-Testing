#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and ADX regime filter
# - Long when price breaks above Camarilla H3 AND 1d volume > 1.3x 20-period average AND ADX(14) > 25
# - Short when price breaks below Camarilla L3 AND 1d volume > 1.3x 20-period average AND ADX(14) > 25
# - Exit when price crosses Camarilla H4/L4 levels (strong reversal)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Camarilla levels provide mathematical support/resistance; volume confirms institutional interest
# - ADX filter ensures we trade only in trending markets, avoiding choppy conditions
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)

name = "4h_1d_camarilla_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h Camarilla pivot levels (based on previous day)
    def calculate_camarilla(h_prev, l_prev, c_prev):
        # Camarilla formula: H4 = C + ((H-L)*1.1/2), H3 = C + ((H-L)*1.1/4)
        # L3 = C - ((H-L)*1.1/4), L4 = C - ((H-L)*1.1/2)
        range_val = h_prev - l_prev
        h4 = c_prev + (range_val * 1.1 / 2)
        h3 = c_prev + (range_val * 1.1 / 4)
        l3 = c_prev - (range_val * 1.1 / 4)
        l4 = c_prev - (range_val * 1.1 / 2)
        return h3, l3, h4, l4
    
    # Get previous day's OHLC for each 4h bar (need to align properly)
    # Since we're on 4h timeframe, we use daily OHLC from previous completed day
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    
    # For each bar, use previous day's OHLC (we'll compute this properly)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_h3_1d = np.full_like(high_1d, np.nan)
    camarilla_l3_1d = np.full_like(high_1d, np.nan)
    camarilla_h4_1d = np.full_like(high_1d, np.nan)
    camarilla_l4_1d = np.full_like(high_1d, np.nan)
    
    for i in range(len(high_1d)):
        h3, l3, h4, l4 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        camarilla_h3_1d[i] = h3
        camarilla_l3_1d[i] = l3
        camarilla_h4_1d[i] = h4
        camarilla_l4_1d[i] = l4
    
    # Align daily Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    
    # Pre-compute 4h ADX (14-period) for regime filter
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[high[0] - low[0]], tr])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                          np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                           np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smooth TR, DM+, DM- using Wilder's smoothing (EMA with alpha=1/period)
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            # First value is simple average
            if len(data) >= period:
                result[period-1] = np.mean(data[:period])
                for i in range(period, len(data)):
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result
        
        tr_smooth = wilder_smooth(tr, period)
        dm_plus_smooth = wilder_smooth(dm_plus, period)
        dm_minus_smooth = wilder_smooth(dm_minus, period)
        
        # DI+ and DI-
        di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
        di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
        
        # DX
        dx = np.where((di_plus + di_minus) != 0, 
                     100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
        
        # ADX (smoothed DX)
        adx = wilder_smooth(dx, period)
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    def rolling_mean(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result
    
    vol_ma_1d = rolling_mean(volume_1d, 20)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(adx[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (1d volume > 1.3x 20-period average)
        vol_spike = volume_1d_aligned[i] > 1.3 * vol_ma_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Camarilla H3 breakout AND volume spike AND ADX > 25 (trending)
            if (close[i] > camarilla_h3_aligned[i] and vol_spike and adx[i] > 25):
                position = 1
                signals[i] = 0.25
            # Short conditions: Camarilla L3 breakdown AND volume spike AND ADX > 25 (trending)
            elif (close[i] < camarilla_l3_aligned[i] and vol_spike and adx[i] > 25):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses Camarilla H4/L4 (strong reversal)
            exit_long = (position == 1 and close[i] < camarilla_l4_aligned[i])
            exit_short = (position == -1 and close[i] > camarilla_h4_aligned[i])
            
            # Optional: exit if ADX drops below 20 (trend weakening)
            exit_weak_trend = (adx[i] < 20)
            
            if exit_long or exit_short or exit_weak_trend:
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

def calculate_adx(high, low, close, period=14):
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high[0] - low[0]], tr])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                      np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                       np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    tr_smooth = wilder_smooth(tr, period)
    dm_plus_smooth = wilder_smooth(dm_plus, period)
    dm_minus_smooth = wilder_smooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX
    dx = np.where((di_plus + di_minus) != 0, 
                 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    # ADX (smoothed DX)
    adx = wilder_smooth(dx, period)
    return adx