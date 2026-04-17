# 6h_Aroon_Trend_Filter
# Aroon oscillator (25) + 12h ADX (20) + volume spike.
# Aroon identifies trend strength, ADX confirms trend presence, volume validates.
# Works in bull/bear by capturing strong trends with confirmation.
# Target: 50-150 total trades (12-37/year).

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
    
    # === Aroon oscillator on 6h (25-period) ===
    def calculate_aroon(high, low, period=25):
        aroon_up = np.full(len(high), np.nan)
        aroon_down = np.full(len(high), np.nan)
        for i in range(len(high)):
            if i >= period - 1:
                # Periods since highest high
                highest_high_idx = i - np.argmax(high[i - period + 1:i + 1])
                periods_since_high = (period - 1) - (i - highest_high_idx)
                aroon_up[i] = ((period - 1 - periods_since_high) / (period - 1)) * 100
                
                # Periods since lowest low
                lowest_low_idx = i - np.argmin(low[i - period + 1:i + 1])
                periods_since_low = (period - 1) - (i - lowest_low_idx)
                aroon_down[i] = ((period - 1 - periods_since_low) / (period - 1)) * 100
        return aroon_up - aroon_down  # Oscillator: -100 to +100
    
    aroon_osc = calculate_aroon(high, low, 25)
    
    # === 12h ADX (20-period) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    def calculate_adx(high, low, close, period=20):
        # True Range
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Directional Movement
        dm_plus = np.zeros_like(high)
        dm_minus = np.zeros_like(high)
        for i in range(1, len(high)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            dm_plus[i] = up_move if up_move > down_move and up_move > 0 else 0
            dm_minus[i] = down_move if down_move > up_move and down_move > 0 else 0
        
        # Smoothed values
        def smooth_series(raw, period):
            smoothed = np.full_like(raw, np.nan)
            if len(raw) < period:
                return smoothed
            smoothed[period-1] = np.mean(raw[1:period])
            for i in range(period, len(raw)):
                smoothed[i] = (smoothed[i-1] * (period-1) + raw[i]) / period
            return smoothed
        
        tr_smoothed = smooth_series(tr, period)
        dm_plus_smoothed = smooth_series(dm_plus, period)
        dm_minus_smoothed = smooth_series(dm_minus, period)
        
        # Directional Indicators
        di_plus = np.full_like(high, np.nan)
        di_minus = np.full_like(high, np.nan)
        dx = np.full_like(high, np.nan)
        
        for i in range(len(high)):
            if not np.isnan(tr_smoothed[i]) and tr_smoothed[i] > 0:
                di_plus[i] = (dm_plus_smoothed[i] / tr_smoothed[i]) * 100
                di_minus[i] = (dm_minus_smoothed[i] / tr_smoothed[i]) * 100
                dx[i] = (abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])) * 100
        
        # ADX: smoothed DX
        adx = smooth_series(dx, period)
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 20)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === Volume spike (6h) ===
    vol_avg20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_avg20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    warmup = 50  # Sufficient for Aroon and ADX
    
    for i in range(warmup, n):
        if (np.isnan(aroon_osc[i]) or 
            np.isnan(adx_12h_aligned[i]) or
            np.isnan(vol_avg20[i])):
            signals[i] = 0.0
            continue
        
        vol_spike = volume[i] > 2.0 * vol_avg20[i]
        strong_trend = adx_12h_aligned[i] > 20
        
        if position == 0:
            # Long: Aroon up > 50 (uptrend) + strong trend + volume spike
            if aroon_osc[i] > 50 and strong_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Aroon down > 50 (downtrend) + strong trend + volume spike
            elif aroon_osc[i] < -50 and strong_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Aroon down > 50 or trend weakens
            if aroon_osc[i] < -50 or adx_12h_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Aroon up > 50 or trend weakens
            if aroon_osc[i] > 50 or adx_12h_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Aroon_Trend_Filter"
timeframe = "6h"
leverage = 1.0