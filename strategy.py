#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d ADX trend filter and volume confirmation.
# Enters long when Williams %R crosses above -80 (oversold recovery) AND ADX > 20 (trend present).
# Enters short when Williams %R crosses below -20 (overbought breakdown) AND ADX > 20.
# Williams %R identifies reversal points; ADX filters for trending conditions to avoid whipsaws.
# Volume confirmation adds conviction. Designed for low turnover (target: 15-35 trades/year).
# Works in bull markets (buying dips) and bear markets (selling rallies).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams %R calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h Williams %R (14)
    def calculate_williams_r(high, low, close, period=14):
        highest_high = np.zeros_like(high)
        lowest_low = np.zeros_like(high)
        williams_r = np.full_like(high, -50.0, dtype=float)
        
        for i in range(len(high)):
            if i < period - 1:
                # Not enough data, keep previous value or default
                if i > 0:
                    williams_r[i] = williams_r[i-1]
                continue
            start_idx = i - period + 1
            highest_high[i] = np.max(high[start_idx:i+1])
            lowest_low[i] = np.min(low[start_idx:i+1])
            if highest_high[i] - lowest_low[i] != 0:
                williams_r[i] = -100 * (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])
            else:
                williams_r[i] = williams_r[i-1] if i > 0 else -50.0
        return williams_r
    
    williams_r_12h = calculate_williams_r(high_12h, low_12h, close_12h, 14)
    williams_r_12h_aligned = align_htf_to_ltf(prices, df_12h, williams_r_12h)
    
    # Calculate 1d ADX (14)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_dm_sm = np.zeros_like(high)
        minus_dm_sm = np.zeros_like(high)
        plus_dm_sm[period] = np.sum(plus_dm[1:period+1])
        minus_dm_sm[period] = np.sum(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            plus_dm_sm[i] = plus_dm_sm[i-1] - (plus_dm_sm[i-1] / period) + plus_dm[i]
            minus_dm_sm[i] = minus_dm_sm[i-1] - (minus_dm_sm[i-1] / period) + minus_dm[i]
        
        plus_di = np.where(atr != 0, 100 * plus_dm_sm / atr, 0)
        minus_di = np.where(atr != 0, 100 * minus_dm_sm / atr, 0)
        
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume filter: current volume > 1.3 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need sufficient data for Williams %R and ADX
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_12h_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R signals: cross above -80 (long) or below -20 (short)
        wr = williams_r_12h_aligned[i]
        wr_prev = williams_r_12h_aligned[i-1] if i > 0 else wr
        
        wr_cross_above_80 = (wr_prev <= -80) and (wr > -80)
        wr_cross_below_20 = (wr_prev >= -20) and (wr < -20)
        
        # Trend filter: ADX > 20 (trend present)
        trend_present = adx_1d_aligned[i] > 20
        
        # Volume filter
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        
        if position == 0:
            # Long: Williams %R crosses above -80 + trend present + volume
            if wr_cross_above_80 and trend_present and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 + trend present + volume
            elif wr_cross_below_20 and trend_present and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses above -20 (overbought) OR trend weakens
            wr_cross_above_20 = (wr_prev <= -20) and (wr > -20)
            weak_trend = adx_1d_aligned[i] < 15
            if wr_cross_above_20 or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -80 (oversold) OR trend weakens
            wr_cross_below_80 = (wr_prev >= -80) and (wr < -80)
            weak_trend = adx_1d_aligned[i] < 15
            if wr_cross_below_80 or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_ADX20_Volume"
timeframe = "12h"
leverage = 1.0