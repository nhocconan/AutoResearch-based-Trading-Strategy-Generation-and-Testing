#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with daily EMA trend filter and volume spike confirmation.
# Uses Camarilla levels from daily timeframe for precise entry/exit levels.
# Only takes breakouts in direction of daily EMA34 trend with volume confirmation.
# Designed for low frequency (target: 20-40 trades/year) to minimize fee drag.
# Works in bull markets via upward breakouts and bear markets via downward breakouts.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for Camarilla levels and EMA trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels using previous day's data (to avoid look-ahead)
    # Camarilla formulas:
    # H4 = Close + 1.1*(High - Low)/2
    # L4 = Close - 1.1*(High - Low)/2
    # H3 = Close + 1.1*(High - Low)/4
    # L3 = Close - 1.1*(High - Low)/4
    # H2 = Close + 1.1*(High - Low)/6
    # L2 = Close - 1.1*(High - Low)/6
    # H1 = Close + 1.1*(High - Low)/12
    # L1 = Close - 1.1*(High - Low)/12
    # We use the previous day's data, so we shift by 1
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Calculate Camarilla levels
    H4 = prev_close + 1.1 * (prev_high - prev_low) / 2
    L4 = prev_close - 1.1 * (prev_high - prev_low) / 2
    H3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    L3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    
    # Daily EMA34 for trend filter (using previous day's data)
    ema_34 = pd.Series(prev_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_bullish = close_1d > ema_34
    trend_bearish = close_1d < ema_34
    
    # Volume spike filter (24-period on 4h)
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > 2.0 * vol_ma24
    
    # Align indicators to 4-hour timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or
            np.isnan(vol_ma24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above H3 + daily bullish trend + volume spike
            if (close[i] > H3_aligned[i] and 
                trend_bullish_aligned[i] > 0.5 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 + daily bearish trend + volume spike
            elif (close[i] < L3_aligned[i] and 
                  trend_bearish_aligned[i] > 0.5 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price breaks opposite L3/H3 level or trend changes
            if position == 1:
                if (close[i] < L3_aligned[i] or trend_bullish_aligned[i] <= 0.5):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > H3_aligned[i] or trend_bearish_aligned[i] <= 0.5):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_DailyEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0