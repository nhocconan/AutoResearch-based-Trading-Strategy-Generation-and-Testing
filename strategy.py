#!/usr/bin/env python3
name = "6h_ThreeBarReversal_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 24-period average (4 days of 6h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Three-bar reversal pattern detection
    # Bullish: downtrend (2 lower closes) + bullish reversal candle
    # Bearish: uptrend (2 higher closes) + bearish reversal candle
    lower_close_2 = (close < close[1]) & (close[1] < close[2])
    higher_close_2 = (close > close[1]) & (close[1] > close[2])
    
    # Bullish reversal: current close > previous open AND close > midpoint of previous candle
    bullish_reversal = (close > prices['open'].shift(1).values) & (close > (high[1] + low[1]) / 2)
    # Bearish reversal: current close < previous open AND close < midpoint of previous candle
    bearish_reversal = (close < prices['open'].shift(1).values) & (close < (high[1] + low[1]) / 2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24, 2)  # Wait for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish 3-bar reversal + volume spike + daily uptrend
            vol_condition = volume[i] > vol_ma_24[i] * 2.0
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            bullish_setup = lower_close_2[i] and bullish_reversal[i]
            
            if bullish_setup and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: bearish 3-bar reversal + volume spike + daily downtrend
            elif higher_close_2[i] and bearish_reversal[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: bearish reversal or trend change
            bearish_exit = higher_close_2[i] and bearish_reversal[i]
            trend_change = ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]
            
            if bearish_exit or trend_change:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: bullish reversal or trend change
            bullish_exit = lower_close_2[i] and bullish_reversal[i]
            trend_change = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if bullish_exit or trend_change:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h three-bar reversal with daily trend and volume confirmation
# - Three-bar reversal captures short-term exhaustion and momentum shifts
# - Works in both bull (buy bullish reversals in uptrend) and bear (sell bearish reversals in downtrend)
# - Volume spike (2x average) confirms institutional participation in the reversal
# - Daily EMA(34) filter ensures trades align with higher-timeframe trend
# - Exit on opposite reversal or trend change to avoid giving back profits
# - Position size 0.25 targets 15-35 trades/year, avoiding excessive fee drag
# - Pattern-based approach provides edge in ranging and trending markets alike