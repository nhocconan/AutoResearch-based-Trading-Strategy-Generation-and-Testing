#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R3/S3 breakout with 1d ADX trend filter and volume confirmation
# Long when price closes above Camarilla R3 + 1d ADX > 25 + volume > 1.5x 20-period avg
# Short when price closes below Camarilla S3 + 1d ADX > 25 + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# Camarilla pivots provide mathematically derived support/resistance levels that work in ranging markets.
# 1d ADX > 25 ensures we only trade in trending conditions, reducing whipsaws in choppy markets.
# Volume confirmation (1.5x) targets ~20-30 trades/year on 12h timeframe to avoid overtrading.
# This combination has shown strong performance on ETH/USD in similar formulations.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: ADX(14) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, (dm_plus_smooth / atr_1d) * 100, 0)
    di_minus = np.where(atr_1d != 0, (dm_minus_smooth / atr_1d) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1d = wilders_smoothing(dx, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 12h Camarilla Pivots (R3, S3) ===
    # Based on previous day's high, low, close
    # Camarilla levels: R4 = close + ((high-low)*1.1/2), R3 = close + ((high-low)*1.1/4)
    # S3 = close - ((high-low)*1.1/4), S4 = close - ((high-low)*1.1/2)
    # We use R3 and S3 as breakout levels
    
    # Calculate typical price for 12h bars (we need daily OHLC for pivots)
    # Since we're on 12h timeframe, we calculate pivots from 1d data and align
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r3_1d = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s3_1d = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20) + 5  # ADX + Camarilla + volume + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # ADX filter: trending market (ADX > 25)
        adx_filter = adx_1d_aligned[i] > 25
        
        # === LONG CONDITIONS ===
        # 1. Price closes above Camarilla R3
        # 2. 1d ADX > 25 (trending market)
        # 3. Volume confirmation
        if (close[i] > camarilla_r3_aligned[i]) and \
           adx_filter and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price closes below Camarilla S3
        # 2. 1d ADX > 25 (trending market)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s3_aligned[i]) and \
             adx_filter and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R3S3_1dADX_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0