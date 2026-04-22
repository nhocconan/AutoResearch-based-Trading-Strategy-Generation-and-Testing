#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power Index with 1d EMA50 trend filter and volume confirmation.
# Elder Ray measures bullish/bearish power: Bull Power = High - EMA, Bear Power = Low - EMA.
# Long when Bull Power > 0 and Bear Power < 0 (strong bullish momentum) with price above EMA and volume confirmation.
# Short when Bear Power > 0 and Bull Power < 0 (strong bearish momentum) with price below EMA and volume confirmation.
# Uses daily EMA(50) as higher timeframe trend filter to avoid counter-trend trades.
# Designed for 6h timeframe to target 50-150 trades over 4 years (12-37/year).
# Works in bull/bear via multi-timeframe trend alignment and momentum-based signals.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for higher timeframe trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(13) for Elder Ray on 6h data
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Power Index
    bull_power = high - ema_13  # Bull Power = High - EMA
    bear_power = low - ema_13   # Bear Power = Low - EMA
    
    # 1d EMA(50) for higher timeframe trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.8 * vol_ma20  # Moderate threshold for balance
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Strong bullish momentum: Bull Power > 0 and Bear Power < 0
            strong_bullish = (bull_power[i] > 0) and (bear_power[i] < 0)
            # Strong bearish momentum: Bear Power > 0 and Bull Power < 0
            strong_bearish = (bear_power[i] > 0) and (bull_power[i] < 0)
            
            # Long: strong bullish + price above daily EMA + volume spike
            if (strong_bullish and 
                close[i] > ema_50_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: strong bearish + price below daily EMA + volume spike
            elif (strong_bearish and 
                  close[i] < ema_50_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit on weak bullish momentum or price below EMA
                weak_bullish = (bull_power[i] <= 0) or (bear_power[i] >= 0)
                if (weak_bullish or close[i] < ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on weak bearish momentum or price above EMA
                weak_bearish = (bear_power[i] <= 0) or (bull_power[i] >= 0)
                if (weak_bearish or close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dEMA50_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0