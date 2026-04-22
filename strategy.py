#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band squeeze + volume spike + daily ADX trend filter
# Bollinger Band squeeze (low volatility) precedes breakouts. Volume spike confirms breakout strength.
# Daily ADX > 25 ensures we only trade in trending markets, avoiding whipsaws in ranges.
# Designed for 12h timeframe to capture multi-day swings with low frequency (target: 15-25 trades/year).
# Works in both bull and bear markets by filtering for strong trends only.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for BBands and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Bollinger Bands (20, 2) on daily close
    sma20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    bb_width = (upper - lower) / sma20  # Normalized width
    
    # Bollinger Squeeze: width below 20-period mean width
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma
    
    # ADX (14) on daily data
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    # Smooth TR, +DM, -DM
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    # DI and DX
    plus_di = 100 * plus_dm14 / tr14
    minus_di = 100 * minus_dm14 / tr14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    # ADX: smoothed DX
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_trend = adx > 25
    
    # Volume spike (24-period on 12h)
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > 2.0 * vol_ma24
    
    # Align indicators to 12-hour timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze)
    adx_trend_aligned = align_htf_to_ltf(prices, df_1d, adx_trend)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(squeeze_aligned[i]) or np.isnan(adx_trend_aligned[i]) or
            np.isnan(vol_ma24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long/right after squeeze ends with volume spike and trend
            if not squeeze_aligned[i-1] and squeeze_aligned[i] and vol_spike[i] and adx_trend_aligned[i]:
                # Direction: close above/below 20-day SMA
                sma20_aligned = align_htf_to_ltf(prices, df_1d, sma20)
                if close[i] > sma20_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit: squeeze returns or volume drops
            if position == 1:
                if squeeze_aligned[i] or not vol_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if squeeze_aligned[i] or not vol_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Bollinger_Squeeze_Volume_Spike_ADXTrend"
timeframe = "12h"
leverage = 1.0