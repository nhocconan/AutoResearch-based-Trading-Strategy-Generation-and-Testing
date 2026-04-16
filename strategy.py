#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation (volume > 1.5x 20-bar MA) +
# ATR-based stoploss (2.5x ATR) + position size 0.25. Uses 1d ADX > 25 as trend regime filter
# to avoid whipsaws in ranging markets. Donchian breakouts capture momentum in both bull
# and bear markets; volume confirms institutional participation; ADX ensures trending
# environment; ATR stop manages risk. Targets 20-50 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: ADX(14) for trend regime filter ===
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+ (using Wilder's smoothing = EMA with alpha=1/period)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # first value = simple average
        result[period-1] = np.nanmean(data[1:period]) if not np.all(np.isnan(data[1:period])) else np.nan
        # subsequent values = Wilder smoothing
        alpha = 1.0 / period
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = (1 - alpha) * result[i-1] + alpha * data[i]
        return result
    
    atr_1d = WilderSmooth(tr, 14)
    dm_plus_smooth = WilderSmooth(dm_plus, 14)
    dm_minus_smooth = WilderSmooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = WilderSmooth(dx, 14)
    
    # === Primary timeframe (4h) indicators: Donchian(20), Volume MA(20), ATR(14) for stoploss ===
    # Donchian channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-bar MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss calculation
    tr_4h = np.maximum(np.abs(high[1:] - low[:-1]), np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 40  # covers Donchian(20), Volume MA(20), ATR(14), ADX
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_4h[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        price = close[i]
        vol = volume[i]
        donch_high_i = donch_high[i]
        donch_low_i = donch_low[i]
        vol_ma_i = vol_ma[i]
        atr_i = atr_4h[i]
        adx_val = adx_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price < Donchian Low (breakdown) OR stoploss hit
            if (price < donch_low_i) or (price < entry_price - 2.5 * atr_i):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price > Donchian High (breakout) OR stoploss hit
            if (price > donch_high_i) or (price > entry_price + 2.5 * atr_i):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume confirmation: current volume > 1.5x 20-bar volume MA
            vol_confirmed = vol > 1.5 * vol_ma_i
            
            # Trend regime filter: 1d ADX > 25 (trending market)
            trending = adx_val > 25
            
            if vol_confirmed and trending:
                # LONG: price breaks above Donchian High
                if price > donch_high_i:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                
                # SHORT: price breaks below Donchian Low
                elif price < donch_low_i:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Donchian20_VolumeConfirmation_ADXTrendFilter_ATRStop_V1"
timeframe = "4h"
leverage = 1.0