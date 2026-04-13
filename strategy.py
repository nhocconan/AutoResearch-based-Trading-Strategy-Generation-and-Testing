#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1w volume confirmation and 1d ADX regime filter.
    # Long when price breaks above Camarilla H3 (from 1d) with 1w volume spike and ADX > 25 (trending).
    # Short when price breaks below Camarilla L3 with 1w volume spike and ADX > 25.
    # Exit when price returns to Camarilla pivot point.
    # Uses discrete size 0.25 to minimize fee churn. Target: 50-150 trades over 4 years.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and ADX (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for volume confirmation (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period) for regime filter
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr = np.zeros_like(high)
        tr[0] = high[0] - low[0]
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
        
        # Smoothed TR, DM+ , DM- (Wilder's smoothing)
        atr = np.zeros_like(high)
        dmp = np.zeros_like(high)
        dmm = np.zeros_like(high)
        if len(tr) > period:
            atr[period] = np.nansum(tr[1:period+1])
            dmp[period] = np.nansum(dm_plus[1:period+1])
            dmm[period] = np.nansum(dm_minus[1:period+1])
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                dmp[i] = (dmp[i-1] * (period-1) + dm_plus[i]) / period
                dmm[i] = (dmm[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        dip = np.zeros_like(high)
        dim = np.zeros_like(high)
        dx = np.zeros_like(high)
        for i in range(period, len(high)):
            if atr[i] > 0:
                dip[i] = 100 * dmp[i] / atr[i]
                dim[i] = 100 * dmm[i] / atr[i]
                if dip[i] + dim[i] > 0:
                    dx[i] = 100 * abs(dip[i] - dim[i]) / (dip[i] + dim[i])
                else:
                    dx[i] = 0
        
        # ADX (smoothed DX)
        adx = np.zeros_like(high)
        if len(dx) > 2*period:
            adx[2*period] = np.nansum(dx[period:2*period+1]) / period
            for i in range(2*period+1, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    def calculate_camarilla(high, low, close):
        # Camarilla levels calculated from previous day's OHLC
        pivot = (high + low + close) / 3.0
        range_hl = high - low
        # Resistance levels
        R4 = close + range_hl * 1.5000
        R3 = close + range_hl * 1.2500
        R2 = close + range_hl * 1.1666
        R1 = close + range_hl * 1.0833
        # Support levels
        S1 = close - range_hl * 1.0833
        S2 = close - range_hl * 1.1666
        S3 = close - range_hl * 1.2500
        S4 = close - range_hl * 1.5000
        return pivot, R1, R2, R3, R4, S1, S2, S3, S4
    
    # Calculate Camarilla for each 1d bar (using previous day's data)
    camarilla_pivot = np.full_like(close_1d, np.nan)
    camarilla_R3 = np.full_like(close_1d, np.nan)
    camarilla_L3 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        pivot, R1, R2, R3, R4, S1, S2, S3, S4 = calculate_camarilla(
            high_1d[i-1], low_1d[i-1], close_1d[i-1]
        )
        camarilla_pivot[i] = pivot
        camarilla_R3[i] = R3
        camarilla_L3[i] = S3
    
    # Calculate 1w volume mean (20-period) with min_periods
    volume_1w = df_1w['volume'].values
    volume_series = pd.Series(volume_1w)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h timeframe
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    vol_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_pivot_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or
            np.isnan(camarilla_L3_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or np.isnan(vol_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1w volume > 1.5 * 20-period mean (volume spike)
        volume_confirmation = vol_1w_aligned[i] > 1.5 * vol_ma_aligned[i]
        
        # Regime filter: ADX > 25 indicates trending market
        regime_filter = adx_aligned[i] > 25
        
        # Entry conditions: price breaks Camarilla H3/L3 with volume confirmation and trend regime
        long_entry = (close[i] > camarilla_R3_aligned[i] and volume_confirmation and regime_filter)
        short_entry = (close[i] < camarilla_L3_aligned[i] and volume_confirmation and regime_filter)
        
        # Exit conditions: price returns to Camarilla pivot point (mean reversion to equilibrium)
        long_exit = close[i] < camarilla_pivot_aligned[i]
        short_exit = close[i] > camarilla_pivot_aligned[i]
        
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

name = "12h_1d_camarilla_breakout_1w_volume_adx_v1"
timeframe = "12h"
leverage = 1.0