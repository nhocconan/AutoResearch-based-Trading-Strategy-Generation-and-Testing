#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ADX trend strength filter + Bollinger Bands mean reversion + volume confirmation.
# In strong trends (ADX > 25), price tends to revert to the Bollinger Band mean (20-period SMA).
# In weak trends/ranges (ADX < 20), we avoid trading to prevent whipsaw.
# Volume confirmation ensures institutional participation. Target: 20-40 trades/year (80-160 total).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX(14) calculation on daily timeframe
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original indices
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed values using Wilder's smoothing (EMA with alpha=1/period)
        def wilders_smooth(data, period):
            smoothed = np.full_like(data, np.nan)
            if len(data) < period:
                return smoothed
            # First value is simple average
            smoothed[period-1] = np.nanmean(data[1:period])
            # Subsequent values: smoothed[i] = (smoothed[i-1] * (period-1) + data[i]) / period
            for i in range(period, len(data)):
                if not np.isnan(smoothed[i-1]):
                    smoothed[i] = (smoothed[i-1] * (period-1) + data[i]) / period
                else:
                    smoothed[i] = np.nan
            return smoothed
        
        tr_smoothed = wilders_smooth(tr, period)
        plus_dm_smoothed = wilders_smooth(plus_dm, period)
        minus_dm_smoothed = wilders_smooth(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smoothed / tr_smoothed
        minus_di = 100 * minus_dm_smoothed / tr_smoothed
        
        # DX and ADX
        dx = np.full_like(close, np.nan)
        mask = (plus_di + minus_di) > 0
        dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask])
        
        adx = wilders_smooth(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Bollinger Bands on 4h timeframe (20, 2)
    bb_period = 20
    bb_std = 2.0
    sma = np.full(n, np.nan)
    std_dev = np.full(n, np.nan)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    
    for i in range(bb_period-1, n):
        sma[i] = np.mean(close[i-bb_period+1:i+1])
        std_dev[i] = np.std(close[i-bb_period+1:i+1])
        upper_band[i] = sma[i] + bb_std * std_dev[i]
        lower_band[i] = sma[i] - bb_std * std_dev[i]
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(sma[i]) or 
            np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        adx_val = adx_1d_aligned[i]
        sma_val = sma[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        # ADX filter: only trade when trend is strong enough (ADX > 25)
        strong_trend = adx_val > 25
        
        if position == 0:
            # Long: price near lower BB + strong trend + volume confirmation
            if (price <= lower_band[i] * 1.02 and  # within 2% of lower band
                strong_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price near upper BB + strong trend + volume confirmation
            elif (price >= upper_band[i] * 0.98 and  # within 2% of upper band
                  strong_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to SMA or trend weakens
            if (price >= sma_val * 0.995 or  # near SMA
                adx_val < 20):  # trend weakening
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to SMA or trend weakens
            if (price <= sma_val * 1.005 or  # near SMA
                adx_val < 20):  # trend weakening
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_ADX_Bollinger_MeanReversion"
timeframe = "4h"
leverage = 1.0