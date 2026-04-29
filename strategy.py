#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with Daily Trend Filter and Volume Spike
# Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Daily trend filter (price vs EMA34) ensures alignment with higher timeframe trend
# Volume spike confirms institutional participation in the move
# Works in all regimes: captures momentum bursts in both bull and bear markets
# Target: 15-30 trades/year (60-120 total over 4 years)

name = "6h_ElderRay_1dEMA34_Trend_VolumeSpike_v3"
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
    
    # Load HTF data ONCE before loop for daily calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate EMA13 for Elder Ray (6h)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 34, 20, 13)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema34_1d = ema34_1d_aligned[i]
        
        # Determine trend regime from daily EMA34
        bullish_regime = curr_close > curr_ema34_1d
        bearish_regime = curr_close < curr_ema34_1d
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power positive AND volume spike in bullish regime
            if curr_bull_power > 0 and curr_volume_confirm and bullish_regime:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative AND volume spike in bearish regime
            elif curr_bear_power < 0 and curr_volume_confirm and bearish_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position - exit when Bull Power turns negative
            if curr_bull_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit when Bear Power turns positive
            if curr_bear_power >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals