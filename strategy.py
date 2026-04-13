#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with volume confirmation and 12h ADX > 25 trend filter.
    # Works in both bull and bear markets by only taking trades in the direction of the 12h trend.
    # Uses discrete position size 0.25 to minimize fee churn. Target: 75-200 total trades over 4 years.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivots and volume (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 12h data for ADX trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivot levels (based on previous 4h bar)
    def calculate_camarilla(high, low, close):
        # Camarilla levels based on previous bar's range
        # H4 = Close + 1.5 * (High - Low)
        # H3 = Close + 1.25 * (High - Low)
        # H2 = Close + 1.166 * (High - Low)
        # H1 = Close + 1.0833 * (High - Low)
        # L1 = Close - 1.0833 * (High - Low)
        # L2 = Close - 1.166 * (High - Low)
        # L3 = Close - 1.25 * (High - Low)
        # L4 = Close - 1.5 * (High - Low)
        H4 = close + 1.5 * (high - low)
        H3 = close + 1.25 * (high - low)
        H2 = close + 1.166 * (high - low)
        H1 = close + 1.0833 * (high - low)
        L1 = close - 1.0833 * (high - low)
        L2 = close - 1.166 * (high - low)
        L3 = close - 1.25 * (high - low)
        L4 = close - 1.5 * (high - low)
        return H4, H3, H2, H1, L1, L2, L3, L4
    
    # Shift by 1 to use previous bar's levels (no look-ahead)
    H4_4h = np.full_like(df_4h['close'].values, np.nan)
    H3_4h = np.full_like(df_4h['close'].values, np.nan)
    H2_4h = np.full_like(df_4h['close'].values, np.nan)
    H1_4h = np.full_like(df_4h['close'].values, np.nan)
    L1_4h = np.full_like(df_4h['close'].values, np.nan)
    L2_4h = np.full_like(df_4h['close'].values, np.nan)
    L3_4h = np.full_like(df_4h['close'].values, np.nan)
    L4_4h = np.full_like(df_4h['close'].values, np.nan)
    
    for i in range(1, len(df_4h)):
        H4_4h[i], H3_4h[i], H2_4h[i], H1_4h[i], L1_4h[i], L2_4h[i], L3_4h[i], L4_4h[i] = \
            calculate_camarilla(df_4h['high'].values[i-1], df_4h['low'].values[i-1], df_4h['close'].values[i-1])
    
    # Calculate 4h volume mean (20-period) with min_periods
    volume_4h_series = pd.Series(df_4h['volume'].values)
    vol_ma_20_4h = volume_4h_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h ADX (14-period) with min_periods
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Directional Movement
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        for i in range(1, len(high)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            elif down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
        
        # Smoothed TR, +DM, -DM (Wilder's smoothing)
        atr = np.zeros_like(high)
        plus_dm_smooth = np.zeros_like(high)
        minus_dm_smooth = np.zeros_like(high)
        
        if len(tr) > period:
            # Initial values
            atr[period] = np.nansum(tr[1:period+1])
            plus_dm_smooth[period] = np.nansum(plus_dm[1:period+1])
            minus_dm_smooth[period] = np.nansum(minus_dm[1:period+1])
            
            # Wilder's smoothing
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
                minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Directional Indicators
        plus_di = np.full_like(high, np.nan)
        minus_di = np.full_like(high, np.nan)
        dx = np.full_like(high, np.nan)
        
        for i in range(period, len(high)):
            if atr[i] > 0:
                plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
                minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
                if plus_di[i] + minus_di[i] > 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # ADX (smoothed DX)
        adx = np.full_like(high, np.nan)
        if len(dx) > 2*period:
            adx[2*period] = np.nansum(dx[period:2*period+1]) / period
            for i in range(2*period+1, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_12h = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 14)
    
    # Align HTF indicators to 4h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_4h, H4_4h)
    H3_aligned = align_htf_to_ltf(prices, df_4h, H3_4h)
    H2_aligned = align_htf_to_ltf(prices, df_4h, H2_4h)
    H1_aligned = align_htf_to_ltf(prices, df_4h, H1_4h)
    L1_aligned = align_htf_to_ltf(prices, df_4h, L1_4h)
    L2_aligned = align_htf_to_ltf(prices, df_4h, L2_4h)
    L3_aligned = align_htf_to_ltf(prices, df_4h, L3_4h)
    L4_aligned = align_htf_to_ltf(prices, df_4h, L4_4h)
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or np.isnan(vol_ma_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h volume for spike detection
        volume_4h_raw = df_4h['volume'].values
        vol_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h_raw)
        
        # Volume filter: current 4h volume > 1.5 * 20-period mean (volume spike)
        volume_confirmation = vol_4h_aligned[i] > 1.5 * vol_ma_aligned[i]
        
        # Trend filter: 12h ADX > 25 indicates strong trend
        trend_filter = adx_aligned[i] > 25
        
        # Entry conditions: price breaks Camarilla H4 (long) or L4 (short) with volume confirmation and trend
        long_entry = (close[i] > H4_aligned[i] and volume_confirmation and trend_filter)
        short_entry = (close[i] < L4_aligned[i] and volume_confirmation and trend_filter)
        
        # Exit conditions: price returns to Camarilla H3/L3 levels (profit taking)
        long_exit = close[i] < H3_aligned[i]
        short_exit = close[i] > L3_aligned[i]
        
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

name = "4h_4h_12h_camarilla_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0