#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with 1d EMA trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13. Bull Power = High - EMA13, Bear Power = Low - EMA13.
# In trending markets: Bull Power > 0 and rising = long signal, Bear Power < 0 and falling = short signal.
# Uses 1d EMA34 for higher timeframe trend filter (long only above, short only below).
# Volume confirmation ensures breakouts have participation. Works in bull (trend continuation) and bear (mean reversion at extremes).

name = "6h_ElderRay_Power_1dEMA34_Trend_VolumeConfirm_v1"
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
    
    # 1d HTF data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 (using prior day's close to avoid look-ahead)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Shift by 1 to use only completed daily bars
    ema_34_1d_shifted = np.roll(ema_34_1d, 1)
    ema_34_1d_shifted[0] = np.nan  # First value has no prior day
    
    # Align 1d EMA to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_shifted)
    
    # Elder Ray components: EMA13 of close, then Bull/Bear Power
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Smooth the power signals with 5-period EMA to reduce noise
    bull_power_smooth = pd.Series(bull_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 40  # Need sufficient history for EMAs and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # 1d trend filter
        above_1d_trend = close[i] > ema_34_1d_aligned[i]
        below_1d_trend = close[i] < ema_34_1d_aligned[i]
        
        # Elder Ray signals with momentum
        bull_power_rising = bull_power_smooth[i] > bull_power_smooth[i-1]
        bear_power_falling = bear_power_smooth[i] < bear_power_smooth[i-1]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 and rising, above 1d trend, volume confirmation
            if bull_power_smooth[i] > 0 and bull_power_rising and above_1d_trend and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 and falling, below 1d trend, volume confirmation
            elif bear_power_smooth[i] < 0 and bear_power_falling and below_1d_trend and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Bear Power turning positive or trend failure
            if bear_power_smooth[i] > 0 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Bull Power turning negative or trend failure
            if bull_power_smooth[i] < 0 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals