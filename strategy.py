#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Regime Filter with Volume Spike Confirmation
# Uses Bull Power (EMA13 - Low) and Bear Power (High - EMA13) from 6h for momentum,
# 1d ADX regime filter to distinguish trending (ADX>25) from ranging (ADX<20) markets,
# and volume spike confirmation for entry timing. Designed for 12-35 trades/year
# to minimize fee drag while capturing sustained moves in both bull and bear markets.
# Elder Ray identifies institutional buying/selling pressure, ADX regime prevents
# whipsaw in choppy markets, volume spike confirms participation.

name = "6h_ElderRay_1dADXRegime_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period) for regime filter
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # Align with original indices
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed TR, PlusDM, MinusDM (Wilder's smoothing)
        tr_period = len(tr)
        atr = np.full(tr_period, np.nan)
        plus_dm_smooth = np.full(tr_period, np.nan)
        minus_dm_smooth = np.full(tr_period, np.nan)
        
        # Initial values
        if tr_period >= period:
            atr[period-1] = np.nanmean(tr[1:period+1])
            plus_dm_smooth[period-1] = np.nanmean(plus_dm[1:period+1])
            minus_dm_smooth[period-1] = np.nanmean(minus_dm[1:period+1])
            
            # Wilder's smoothing
            for i in range(period, tr_period):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
                minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Directional Indicators
        plus_di = np.full(tr_period, np.nan)
        minus_di = np.full(tr_period, np.nan)
        dx = np.full(tr_period, np.nan)
        
        for i in range(period, tr_period):
            if atr[i] != 0:
                plus_di[i] = (plus_dm_smooth[i] / atr[i]) * 100
                minus_di[i] = (minus_dm_smooth[i] / atr[i]) * 100
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        
        # ADX (smoothed DX)
        adx = np.full(tr_period, np.nan)
        if tr_period >= 2*period-1:
            adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
            for i in range(2*period-1, tr_period):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h EMA13 for Elder Ray
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = ema13_6h - low  # Buying power: EMA13 minus low
    bear_power = high - ema13_6h  # Selling power: high minus EMA13
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema13_6h[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filters: ADX > 25 = trending, ADX < 20 = ranging
        is_trending = adx_1d_aligned[i] > 25
        is_ranging = adx_1d_aligned[i] < 20
        
        if position == 0:
            # Long conditions: Bull Power > 0 (buying pressure) AND volume spike
            # In trending markets: follow momentum
            # In ranging markets: mean reversion from extreme Bear Power
            if ((is_trending and bull_power[i] > 0 and volume[i] > (2.0 * vol_ema_20[i])) or
                (is_ranging and bull_power[i] > 0 and bear_power[i] < np.percentile(bear_power[max(0,i-50):i+1], 20) and volume[i] > (2.0 * vol_ema_20[i]))):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power > 0 (selling pressure) AND volume spike
            # In trending markets: follow momentum
            # In ranging markets: mean reversion from extreme Bull Power
            elif ((is_trending and bear_power[i] > 0 and volume[i] > (2.0 * vol_ema_20[i])) or
                  (is_ranging and bear_power[i] > 0 and bull_power[i] > np.percentile(bull_power[max(0,i-50):i+1], 80) and volume[i] > (2.0 * vol_ema_20[i]))):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power > 0 (selling pressure takes over) OR loss of momentum
            if bear_power[i] > 0 or (is_trending and bull_power[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power > 0 (buying pressure takes over) OR loss of momentum
            if bull_power[i] > 0 or (is_trending and bear_power[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals