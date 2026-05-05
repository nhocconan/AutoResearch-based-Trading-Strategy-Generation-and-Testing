#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator (Jaw/Teeth/Lips) + 1d EMA50 trend filter + volume confirmation
# Williams Alligator: Jaw=SMA(13,8), Teeth=SMA(8,5), Lips=SMA(5,3)
# Long when: Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA50 AND volume > 1.5x 20-period MA
# Short when: Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA50 AND volume > 1.5x 20-period MA
# Exit when: Alligator alignment reverses OR volume < 1.2x 20-period MA (loss of conviction)
# Uses Alligator for trend alignment, 1d EMA for higher-timeframe trend filter, volume for conviction
# Timeframe: 4h, HTF: 1d for EMA50. Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

name = "4h_WilliamsAlligator_1dEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator on 4h
    # Jaw: SMA(13,8) - 13-period SMA, 8-period shift
    if len(close) >= 13:
        jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
        jaw = np.concatenate([np.full(8, np.nan), jaw_raw[:-8]])  # shift 8 periods
    else:
        jaw = np.full(n, np.nan)
    
    # Teeth: SMA(8,5) - 8-period SMA, 5-period shift
    if len(close) >= 8:
        teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean().values
        teeth = np.concatenate([np.full(5, np.nan), teeth_raw[:-5]])  # shift 5 periods
    else:
        teeth = np.full(n, np.nan)
    
    # Lips: SMA(5,3) - 5-period SMA, 3-period shift
    if len(close) >= 5:
        lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean().values
        lips = np.concatenate([np.full(3, np.nan), lips_raw[:-3]])  # shift 3 periods
    else:
        lips = np.full(n, np.nan)
    
    # Alligator alignment signals
    bullish_alignment = (lips > teeth) & (teeth > jaw)  # Lips > Teeth > Jaw
    bearish_alignment = (lips < teeth) & (teeth < jaw)  # Lips < Teeth < Jaw
    
    # Volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
        volume_exit = volume > (1.2 * vol_ma_20)  # softer exit condition
    else:
        volume_filter = np.zeros(n, dtype=bool)
        volume_exit = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d
    close_1d = df_1d['close'].values
    if len(close_1d) >= 50:
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    else:
        ema_50_1d = np.full(len(df_1d), np.nan)
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bullish_alignment[i]) or np.isnan(bearish_alignment[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(volume_exit[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: bullish alignment + price > 1d EMA50 + volume filter
            if (bullish_alignment[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish alignment + price < 1d EMA50 + volume filter
            elif (bearish_alignment[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: alignment reverses OR volume weakens
            if (not bullish_alignment[i] or not volume_exit[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: alignment reverses OR volume weakens
            if (not bearish_alignment[i] or not volume_exit[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals