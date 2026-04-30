#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d ADX trend filter and volume confirmation
# Bollinger Band squeeze (low volatility) precedes explosive moves. Breakout direction 
# filtered by 1d ADX > 25 (trending market) and volume spike (2.0x 20-period average).
# Works in both bull and bear markets by taking breakout direction only when 
# 1d ADX confirms trending conditions. Discrete sizing 0.25 minimizes fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_BollingerSqueeze_Breakout_1dADX_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: smoothed = previous_smoothed * (1 - 1/period) + current * (1/period)
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]):
                    result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
        return result
    
    period_adx = 14
    atr = wilders_smoothing(tr, period_adx)
    plus_di = 100 * wilders_smoothing(plus_dm, period_adx) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, period_adx) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period_adx)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean()
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std()
    upper_band = sma_20 + (bb_std * std_20)
    lower_band = sma_20 - (bb_std * std_20)
    bb_width = (upper_band - lower_band) / sma_20
    
    # Bollinger Band squeeze: width below 20-period average width
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean()
    squeeze_condition = bb_width < bb_width_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 20, 34)  # warmup
    
    for i in range(start_idx, n):
        # Get 1d ADX value (prior completed 1d bar)
        curr_adx = adx_1d_aligned[i]
        
        # Bollinger Band conditions
        bb_squeeze = squeeze_condition[i]
        breakout_up = close[i] > upper_band[i]
        breakout_down = close[i] < lower_band[i]
        
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (2.0 * vol_ma_20)
        
        if position == 0:  # Flat - look for new entries
            # Require ADX > 25 (trending market), volume spike, and Bollinger squeeze breakout
            if curr_adx > 25 and volume_spike and bb_squeeze:
                if breakout_up:
                    signals[i] = 0.25
                    position = 1
                elif breakout_down:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price returns to middle Bollinger Band (mean reversion) or ADX weakens
            if close[i] < sma_20[i] or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price returns to middle Bollinger Band (mean reversion) or ADX weakens
            if close[i] > sma_20[i] or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals