#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Long: Bull Power > 0 AND Bear Power < 0 AND price > 1d EMA34 (uptrend) AND volume > 1.5x average
# Short: Bear Power < 0 AND Bull Power > 0 AND price < 1d EMA34 (downtrend) AND volume > 1.5x average
# Exit: Opposite Elder Ray signal OR price crosses 1d EMA34
# Designed to work in both bull and bear markets by combining trend filter with momentum
# Target: 12-37 trades/year via tight Elder Ray conditions + volume + trend filter

name = "6h_ElderRay_BullBearPower_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe (completed 1d candles only)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Elder Ray components on 6h timeframe
    # Bull Power = High - EMA(13) of close
    # Bear Power = Low - EMA(13) of close
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power
    bear_power = low - ema13   # Bear Power
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 34, 13)  # Need sufficient history for volume MA, EMA34, and EMA13
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_ma_20[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        ema34_val = ema34_1d_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 AND Bear Power < 0 AND uptrend AND volume spike
            if bull > 0 and bear < 0 and price > ema34_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Bear Power < 0 AND Bull Power > 0 AND downtrend AND volume spike
            elif bear < 0 and bull > 0 and price < ema34_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on opposite Elder Ray signal or price < EMA34
            # Exit: Bear Power >= 0 (bullish momentum fading) OR price < EMA34 (trend broken)
            if bear >= 0 or price < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on opposite Elder Ray signal or price > EMA34
            # Exit: Bull Power <= 0 (bearish momentum fading) OR price > EMA34 (trend broken)
            if bull <= 0 or price > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals