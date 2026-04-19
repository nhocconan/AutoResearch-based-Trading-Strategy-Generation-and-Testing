#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with daily volume confirmation and 1d ADX filter.
# Long when price breaks above R1 (4/9 pivot level) AND volume > 1.3x daily average volume AND ADX(14) > 20 (trending regime)
# Short when price breaks below S1 (5/9 pivot level) AND volume > 1.3x daily average volume AND ADX(14) > 20
# Exit when price crosses back through the pivot point (central level)
# Uses Camarilla for precision levels, volume for confirmation, ADX to avoid chop.
# Target: 20-30 trades/year per symbol.
name = "4h_Camarilla_R1S1_Volume_ADX"
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
    
    # Get 1d OHLC for Camarilla pivot and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12, PP = (high+low+close)/3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels (use previous day's data)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First day will have NaN due to roll, handled later
    
    # Camarilla levels
    R1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    S1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    PP = (prev_high + prev_low + prev_close) / 3
    
    # Calculate ADX(14) on 1d
    # ADX requires +DI and -DI
    # +DM = max(0, high - prev_high) if high - prev_high > prev_low - low else 0
    # -DM = max(0, prev_low - low) if prev_low - low > high - prev_high else 0
    high_diff = high_1d - prev_high
    low_diff = prev_low - low_1d
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - prev_close)
    tr3 = np.abs(low_1d - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) >= period:
            # First value is simple average
            result[period-1] = np.nansum(arr[:period])
            # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
            for i in range(period, len(arr)):
                if not np.isnan(result[i-1]) and not np.isnan(arr[i]):
                    result[i] = result[i-1] * (1 - 1/period) + arr[i] * (1/period)
        return result
    
    period = 14
    atr_1d = wilder_smooth(tr, period)
    plus_di_1d = 100 * wilder_smooth(plus_dm, period) / atr_1d
    minus_di_1d = 100 * wilder_smooth(minus_dm, period) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilder_smooth(dx_1d, period)
    
    # Get 1d average volume for confirmation
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d arrays to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Ensure indicators are ready (20 for roll, 34 for ADX with smoothing)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(PP_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx_1d_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        r1 = R1_aligned[i]
        s1 = S1_aligned[i]
        pp = PP_aligned[i]
        
        # Regime filter: only trade in trending market (ADX > 20)
        trending_regime = adx_val > 20
        
        if position == 0:
            # Long entry: break above R1 + volume spike + trending regime
            if price > r1 and vol > 1.3 * vol_ma and trending_regime:
                signals[i] = 0.25
                position = 1
            # Short entry: break below S1 + volume spike + trending regime
            elif price < s1 and vol > 1.3 * vol_ma and trending_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below pivot point
            if price < pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above pivot point
            if price > pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals