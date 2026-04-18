#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily breakout strategy using weekly high/low channels with volume confirmation.
# Uses weekly Donchian channel (20-week high/low) for trend context and daily price action
# for entry timing. Enters on daily breakouts above weekly high or below weekly low
# with volume > 1.5x 20-day average. Exits on opposite weekly channel touch or ATR stop.
# Designed for low frequency (target 15-25 trades/year) to minimize fee drag.
# Works in bull markets (breakouts continue) and bear markets (mean reversion at extremes).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channel (20-period)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channel (20-period high/low)
    donchian_high_20w = np.full(len(high_1w), np.nan)
    donchian_low_20w = np.full(len(low_1w), np.nan)
    for i in range(20, len(high_1w)):
        donchian_high_20w[i] = np.max(high_1w[i-20:i])
        donchian_low_20w[i] = np.min(low_1w[i-20:i])
    
    # Align weekly Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20w)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20w)
    
    # Calculate daily ATR(14) for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_daily = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-day volume moving average for confirmation
    vol_ma_20d = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20d[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # need weekly Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_daily[i]) or np.isnan(vol_ma_20d[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-day average
        vol_confirmed = volume[i] > 1.5 * vol_ma_20d[i]
        
        if position == 0:
            # Long entry: daily close breaks above weekly Donchian high with volume
            if (close[i] > donchian_high_aligned[i] and vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short entry: daily close breaks below weekly Donchian low with volume
            elif (close[i] < donchian_low_aligned[i] and vol_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price touches weekly Donchian low or ATR stop
            if close[i] <= donchian_low_aligned[i] or close[i] < open_price[i] - 2.0 * atr_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches weekly Donchian high or ATR stop
            if close[i] >= donchian_high_aligned[i] or close[i] > open_price[i] + 2.0 * atr_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian20_Breakout_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0