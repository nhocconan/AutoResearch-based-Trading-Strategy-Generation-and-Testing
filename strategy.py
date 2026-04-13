#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h price action with 1d volume-weighted price trend and volume confirmation.
# Uses 1d VWAP trend filter to identify institutional flow direction, combined with
# 12h price rejection at VWAP (close crossing VWAP with volume) for entries.
# VWAP acts as dynamic support/resistance - price tends to revert to it in ranging
# markets and break through it in trending markets with volume.
# In ranging markets (low ADX): fade VWAP crosses (mean reversion).
# In trending markets (high ADX): break through VWAP with volume (trend follow).
# ADX regime filter prevents whipsaws. Designed for 12h timeframe to minimize trades.
# Works in both bull/bear by adapting to regime - mean reversion in range, trend follow in trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for VWAP trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1-day VWAP (typical price * volume)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_num = np.cumsum(typical_price_1d * volume_1d)
    vwap_den = np.cumsum(volume_1d)
    vwap_1d = np.where(vwap_den != 0, vwap_num / vwap_den, typical_price_1d)
    
    # Align 1d VWAP to 12h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # 14-period ADX for regime filtering (trending vs ranging)
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smoothed values
        atr = np.full_like(tr, np.nan)
        dm_plus_smooth = np.full_like(dm_plus, np.nan)
        dm_minus_smooth = np.full_like(dm_minus, np.nan)
        
        # First values (simple average)
        if len(tr) >= period:
            atr[period-1] = np.nanmean(tr[1:period])
            dm_plus_smooth[period-1] = np.nanmean(dm_plus[1:period])
            dm_minus_smooth[period-1] = np.nanmean(dm_minus[1:period])
        
        # Wilder's smoothing
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        plus_di = 100 * dm_plus_smooth / atr
        minus_di = 100 * dm_minus_smooth / atr
        
        # DX and ADX
        dx = np.full_like(close, np.nan)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        
        adx = np.full_like(close, np.nan)
        if len(dx) >= 2*period-1:
            adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
            for i in range(2*period-1, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Average volume for confirmation (20-period)
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vwap = vwap_1d_aligned[i]
        adx = adx_1d_aligned[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # In ranging market (ADX < 25): mean reversion at VWAP
            # In trending market (ADX >= 25): trend follow through VWAP
            if adx < 25:
                # Ranging: fade VWAP crosses
                # Long: price crosses below VWAP with volume (oversold bounce)
                if close[i-1] >= vwap and price < vwap and volume_confirm:
                    position = 1
                    signals[i] = position_size
                # Short: price crosses above VWAP with volume (overbought rejection)
                elif close[i-1] <= vwap and price > vwap and volume_confirm:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                # Trending: break through VWAP with volume
                # Long: price breaks above VWAP with volume (continuation)
                if close[i-1] <= vwap and price > vwap and volume_confirm:
                    position = 1
                    signals[i] = position_size
                # Short: price breaks below VWAP with volume (continuation)
                elif close[i-1] >= vwap and price < vwap and volume_confirm:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to VWAP or adverse VWAP cross
            if price >= vwap or (close[i-1] < vwap and price > vwap):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to VWAP or adverse VWAP cross
            if price <= vwap or (close[i-1] > vwap and price < vwap):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_VWAP_Trend_MeanReversion"
timeframe = "12h"
leverage = 1.0