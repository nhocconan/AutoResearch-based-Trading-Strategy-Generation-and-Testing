#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation and 1d ADX trend filter.
    # Long when price breaks above Camarilla R4 with 1d volume spike and 1d ADX > 25.
    # Short when price breaks below Camarilla S4 with 1d volume spike and 1d ADX > 25.
    # Exit when price returns to Camarilla H5/L5 levels (mean reversion).
    # Uses discrete size 0.25 to minimize fee churn. Target: 75-150 trades over 4 years.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots, volume confirmation and ADX trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivots (based on previous day's OHLC)
    def calculate_camarilla(high, low, close):
        # Camarilla levels based on previous day's range
        range_val = high - low
        camarilla_h5 = close + range_val * 1.1 / 2
        camarilla_h4 = close + range_val * 1.1 / 4
        camarilla_h3 = close + range_val * 1.1 / 6
        camarilla_l3 = close - range_val * 1.1 / 6
        camarilla_l4 = close - range_val * 1.1 / 4
        camarilla_l5 = close - range_val * 1.1 / 2
        return camarilla_h5, camarilla_h4, camarilla_h3, camarilla_l3, camarilla_l4, camarilla_l5
    
    # Calculate for previous day (shift by 1 to avoid look-ahead)
    camarilla_h5 = np.full_like(df_1d['close'].values, np.nan)
    camarilla_h4 = np.full_like(df_1d['close'].values, np.nan)
    camarilla_h3 = np.full_like(df_1d['close'].values, np.nan)
    camarilla_l3 = np.full_like(df_1d['close'].values, np.nan)
    camarilla_l4 = np.full_like(df_1d['close'].values, np.nan)
    camarilla_l5 = np.full_like(df_1d['close'].values, np.nan)
    
    for i in range(1, len(df_1d)):
        h5, h4, h3, l3, l4, l5 = calculate_camarilla(
            df_1d['high'].values[i-1],
            df_1d['low'].values[i-1],
            df_1d['close'].values[i-1]
        )
        camarilla_h5[i] = h5
        camarilla_h4[i] = h4
        camarilla_h3[i] = h3
        camarilla_l3[i] = l3
        camarilla_l4[i] = l4
        camarilla_l5[i] = l5
    
    # Calculate 1d volume mean (20-period) with min_periods
    volume_series = pd.Series(df_1d['volume'].values)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX (14-period) with min_periods
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
    
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Align HTF indicators to 6h timeframe
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Align raw 1d volume for volume spike detection
    volume_1d_raw = df_1d['volume'].values
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_raw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(camarilla_h5_aligned[i]) or np.isnan(camarilla_l5_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 1.5 * 20-period mean (volume spike)
        volume_confirmation = vol_1d_aligned[i] > 1.5 * vol_ma_aligned[i]
        
        # Trend filter: 1d ADX > 25 indicates strong trend
        trend_filter = adx_aligned[i] > 25
        
        # Entry conditions: price breaks Camarilla R4/S4 with volume confirmation and trend
        long_entry = (close[i] > camarilla_h4_aligned[i] and volume_confirmation and trend_filter)
        short_entry = (close[i] < camarilla_l4_aligned[i] and volume_confirmation and trend_filter)
        
        # Exit conditions: price returns to Camarilla H5/L5 levels (mean reversion)
        long_exit = close[i] < camarilla_h5_aligned[i]
        short_exit = close[i] > camarilla_l5_aligned[i]
        
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

name = "6h_1d_camarilla_breakout_volume_adx_v1"
timeframe = "6h"
leverage = 1.0