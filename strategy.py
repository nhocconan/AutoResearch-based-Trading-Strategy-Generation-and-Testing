#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian(20) breakout from 12h HTF + volume confirmation + ATR trailing stop
# Long when price breaks above 12h Donchian high(20) AND volume > 1.5 * avg_volume(20) AND ATR(14) < 0.03 * close (low volatility filter)
# Short when price breaks below 12h Donchian low(20) AND volume > 1.5 * avg_volume(20) AND ATR(14) < 0.03 * close
# Exit via ATR trailing stop: 3 * ATR(14) from highest high (long) or lowest low (short)
# Uses discrete sizing 0.30 to balance return and drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 12h Donchian provides clear structure with proven breakout edge
# Volume confirmation filters weak breakouts (reduces false signals)
# Low volatility filter ensures entries during consolidation before expansion
# ATR trailing stop manages risk without look-ahead

name = "4h_12hDonchian20_Breakout_VolumeLowVol_ATRTrail_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:  # Need sufficient data for Donchian(20)
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    high_series_12h = pd.Series(high_12h)
    low_series_12h = pd.Series(low_12h)
    donchian_high_12h = high_series_12h.rolling(window=20, min_periods=20).max().values
    donchian_low_12h = low_series_12h.rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for volatility filter and trailing stop
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = 0  # First bar has no previous close
    tr3.iloc[0] = 0
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 12h Donchian levels and ATR to 4h timeframe (wait for completed 12h bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_12h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_12h)
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Low volatility filter: ATR(14) < 3% of close price
    low_vol_filter = atr < (0.03 * close)
    low_vol_aligned = align_htf_to_ltf(prices, df_12h, low_vol_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high = 0.0
    lowest_low = np.inf
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(avg_volume_20[i]) or np.isnan(low_vol_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high = 0.0
                lowest_low = np.inf
            continue
        
        if position == 0:
            # Long: price breaks above 12h Donchian high with volume confirmation and low volatility
            if (close[i] > donchian_high_aligned[i] and close[i-1] <= donchian_high_aligned[i-1] and 
                volume_confirm[i] and low_vol_aligned[i]):
                signals[i] = 0.30
                position = 1
                highest_high = close[i]
            # Short: price breaks below 12h Donchian low with volume confirmation and low volatility
            elif (close[i] < donchian_low_aligned[i] and close[i-1] >= donchian_low_aligned[i-1] and 
                  volume_confirm[i] and low_vol_aligned[i]):
                signals[i] = -0.30
                position = -1
                lowest_low = close[i]
        elif position == 1:
            # Update highest high for trailing stop
            highest_high = max(highest_high, close[i])
            # Exit long: price drops 3*ATR below highest high (trailing stop)
            if close[i] < highest_high - 3.0 * atr_aligned[i]:
                signals[i] = 0.0
                position = 0
                highest_high = 0.0
                lowest_low = np.inf
            else:
                signals[i] = 0.30
        elif position == -1:
            # Update lowest low for trailing stop
            lowest_low = min(lowest_low, close[i])
            # Exit short: price rises 3*ATR above lowest low (trailing stop)
            if close[i] > lowest_low + 3.0 * atr_aligned[i]:
                signals[i] = 0.0
                position = 0
                highest_high = 0.0
                lowest_low = np.inf
            else:
                signals[i] = -0.30
    
    return signals