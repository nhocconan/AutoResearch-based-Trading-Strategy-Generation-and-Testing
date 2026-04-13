#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and ADX regime filter.
    # Long when price breaks above Camarilla H3 with volume spike and ADX > 20 (trending).
    # Short when price breaks below Camarilla L3 with volume spike and ADX > 20.
    # Exit when price returns to Camarilla H4/L4 levels (pivot-based mean reversion).
    # Uses discrete size 0.25 to minimize fee churn. Target: 75-200 trades over 4 years.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume mean (20-period) with min_periods
    volume_series = pd.Series(volume_1d)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX (14-period) for regime filter
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.nansum(tr[1:period+1]) if period < len(tr) else 0
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        dx = np.zeros_like(close)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align HTF indicators to 4h timeframe
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1d Camarilla pivot levels
    def calculate_camarilla(high, low, close):
        # Camarilla pivot levels based on previous day's range
        pivot = (high + low + close) / 3.0
        range_val = high - low
        
        H4 = pivot + (range_val * 1.1 / 2)
        H3 = pivot + (range_val * 1.1 / 4)
        H2 = pivot + (range_val * 1.1 / 6)
        H1 = pivot + (range_val * 1.1 / 12)
        
        L1 = pivot - (range_val * 1.1 / 12)
        L2 = pivot - (range_val * 1.1 / 6)
        L3 = pivot - (range_val * 1.1 / 4)
        L4 = pivot - (range_val * 1.1 / 2)
        
        return H4, H3, H2, H1, L1, L2, L3, L4
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_H4 = np.zeros_like(close_1d)
    camarilla_H3 = np.zeros_like(close_1d)
    camarilla_H2 = np.zeros_like(close_1d)
    camarilla_H1 = np.zeros_like(close_1d)
    camarilla_L1 = np.zeros_like(close_1d)
    camarilla_L2 = np.zeros_like(close_1d)
    camarilla_L3 = np.zeros_like(close_1d)
    camarilla_L4 = np.zeros_like(close_1d)
    
    for i in range(len(close_1d)):
        if i == 0:
            # For first bar, use same values (will be aligned properly)
            camarilla_H4[i] = camarilla_H3[i] = camarilla_H2[i] = camarilla_H1[i] = \
            camarilla_L1[i] = camarilla_L2[i] = camarilla_L3[i] = camarilla_L4[i] = close_1d[i]
        else:
            H4, H3, H2, H1, L1, L2, L3, L4 = calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d[i-1])
            camarilla_H4[i] = H4
            camarilla_H3[i] = H3
            camarilla_H2[i] = H2
            camarilla_H1[i] = H1
            camarilla_L1[i] = L1
            camarilla_L2[i] = L2
            camarilla_L3[i] = L3
            camarilla_L4[i] = L4
    
    # Align Camarilla levels to 4h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    H2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H2)
    H1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H1)
    L1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L1)
    L2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L2)
    L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume (aligned)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        
        # Volume filter: current 1d volume > 1.5 * 20-period mean (volume spike)
        volume_confirmation = vol_1d_aligned[i] > 1.5 * vol_ma_aligned[i]
        
        # Regime filter: ADX > 20 indicates trending market (avoid strong ranging)
        regime_filter = adx_aligned[i] > 20
        
        # Entry conditions: price breaks Camarilla H3/L3 levels with volume confirmation and trend regime
        long_entry = (close[i] > H3_aligned[i] and volume_confirmation and regime_filter)
        short_entry = (close[i] < L3_aligned[i] and volume_confirmation and regime_filter)
        
        # Exit conditions: price returns to Camarilla H4/L4 levels (stronger reversal signals)
        long_exit = close[i] < H4_aligned[i]
        short_exit = close[i] > L4_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0