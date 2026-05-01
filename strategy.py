#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and volume confirmation.
# Long when Bull Power > 0 AND price > 1d EMA50 AND volume > 1.5x 20-bar average.
# Short when Bear Power < 0 AND price < 1d EMA50 AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 50-150 total trades over 4 years (12-37/year).
# Elder Ray measures bull/bear strength relative to EMA13, providing clearer momentum signals than price alone.
# 1d EMA50 filter ensures alignment with higher timeframe trend, reducing counter-trend trades.
# Volume confirmation set to 1.5x to avoid choppy market noise while capturing genuine momentum bursts.
# Primary timeframe: 6h, HTF: 1d for trend and EMA calculation.

name = "6h_ElderRay_BullBearPower_1dEMA50_Trend_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA13 and EMA50 on 1d data
    close_1d = pd.Series(df_1d['close'].values)
    ema13_1d = close_1d.ewm(span=13, adjust=False, min_periods=13).mean().values
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMAs to 6h timeframe
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13_1d_aligned
    bear_power = low - ema13_1d_aligned
    
    # Volume confirmation: current 6h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMAs and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(bull_power[i]) or \
           np.isnan(bear_power[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)  # Volume spike threshold
        
        # Elder Ray signals
        bull_power_val = bull_power[i]
        bear_power_val = bear_power[i]
        
        # Trend filter from 1d EMA50
        uptrend = curr_close > ema50_1d_aligned[i]
        downtrend = curr_close < ema50_1d_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 AND uptrend AND volume confirmation
            if (bull_power_val > 0 and 
                uptrend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND downtrend AND volume confirmation
            elif (bear_power_val < 0 and 
                  downtrend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull Power <= 0 (momentum loss) OR price breaks below 1d EMA50 (trend change)
            if (bull_power_val <= 0 or 
                not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power >= 0 (momentum loss) OR price breaks above 1d EMA50 (trend change)
            if (bear_power_val >= 0 or 
                not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals