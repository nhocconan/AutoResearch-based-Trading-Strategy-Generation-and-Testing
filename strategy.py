#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume spike confirmation
# Elder Ray measures bull/bear strength relative to EMA13. In trending regimes (ADX>25 on 1d),
# we take trades in the direction of the 1d trend only when Elder Ray confirms strength.
# Volume spike ensures conviction. Discrete sizing (0.25) minimizes fee churn.
# Target: 12-37 trades/year per symbol. Works in both bull (trend following) and bear (range/mean reversion via regime).

name = "6h_ElderRay_1dADX25_VolumeSpike_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter and EMA13 for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX(14) + EMA(13)
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d ADX (14-period)
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
    up_move = np.where(high_1d - np.roll(high_1d, 1) > 0, high_1d - np.roll(high_1d, 1), 0)
    down_move = np.where(np.roll(low_1d, 1) - low_1d > 0, np.roll(low_1d, 1) - low_1d, 0)
    up_move[0] = 0
    down_move[0] = 0
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    plus_dm_1d = wilders_smoothing(up_move, 14)
    minus_dm_1d = wilders_smoothing(down_move, 14)
    
    # +DI and -DI
    plus_di_1d = np.where(atr_1d != 0, (plus_dm_1d / atr_1d) * 100, 0)
    minus_di_1d = np.where(atr_1d != 0, (minus_dm_1d / atr_1d) * 100, 0)
    
    # DX and ADX
    dx_1d = np.where((plus_di_1d + minus_di_1d) != 0, 
                     np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d) * 100, 0)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Align 1d indicators to 6h
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Get 6h data for volume EMA(20) for volume confirmation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    vol_6h = df_6h['volume'].values
    vol_ema_20 = pd.Series(vol_6h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_6h, vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Elder Ray for current 6h bar using 1d EMA13
        bull_power = high[i] - ema_13_1d_aligned[i]  # Bull Power: High - EMA13
        bear_power = low[i] - ema_13_1d_aligned[i]   # Bear Power: Low - EMA13
        
        # Volume confirmation: current 6h volume > 2.0 x 20-period EMA
        volume_confirmed = volume[i] > (2.0 * vol_ema_20_aligned[i])
        
        # 1d regime: trending if ADX > 25, ranging if ADX <= 25
        trending_regime = adx_1d_aligned[i] > 25
        ranging_regime = adx_1d_aligned[i] <= 25
        
        # 1d trend direction from price vs EMA13
        bullish_trend = close[i] > ema_13_1d_aligned[i]
        bearish_trend = close[i] < ema_13_1d_aligned[i]
        
        if position == 0:
            # In trending regime: take Elder Ray signals in direction of 1d trend
            if trending_regime:
                # Long: Bull Power > 0 (strong bulls) + volume confirmation + bullish 1d trend
                if bull_power > 0 and volume_confirmed and bullish_trend:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power < 0 (strong bears) + volume confirmation + bearish 1d trend
                elif bear_power < 0 and volume_confirmed and bearish_trend:
                    signals[i] = -0.25
                    position = -1
            # In ranging regime: fade Elder Ray extremes (mean reversion)
            else:
                # Long: Bear Power < 0 (bears exhausted) + volume confirmation
                if bear_power < 0 and volume_confirmed:
                    signals[i] = 0.25
                    position = 1
                # Short: Bull Power > 0 (bulls exhausted) + volume confirmation
                elif bull_power > 0 and volume_confirmed:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Bull Power turns negative OR regime shifts to ranging and Bear Power < 0
            if bull_power <= 0 or (ranging_regime and bear_power < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power turns positive OR regime shifts to ranging and Bull Power > 0
            if bear_power >= 0 or (ranging_regime and bull_power > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals