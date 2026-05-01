#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d Supertrend filter and volume confirmation
# Uses 1d Supertrend (ATR=10, mult=3) as trend filter - only trades in direction of higher timeframe trend
# Volume spike (2.0x 20-period MA) confirms institutional participation
# Works in bull/bear via trend filter - avoids counter-trend whipsaws
# Designed for low frequency (50-150 trades over 4 years) to minimize fee drag on 4h timeframe
# Focus on BTC/ETH - avoids SOL bias by requiring HTF trend alignment

name = "4h_Camarilla_R3S3_Breakout_1dSupertrend_Trend_VolumeSpike_v1"
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
    
    # 1d HTF data for Supertrend trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ATR calculation
        return np.zeros(n)
    
    # 1d Supertrend calculation (ATR=10, mult=3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with indices
    
    # ATR(10) using Wilder's smoothing
    def wilders_smoothing(x, period):
        result = np.full_like(x, np.nan)
        if len(x) >= period:
            first_val = np.nansum(x[1:period+1])  # skip first NaN
            result[period] = first_val
            for i in range(period+1, len(x)):
                result[i] = result[i-1] - (result[i-1] / period) + x[i]
        return result
    
    atr_period = 10
    atr = wilders_smoothing(tr, atr_period)
    
    # Supertrend calculation
    hl2 = (high_1d + low_1d) / 2
    multiplier = 3
    
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    supertrend = np.full_like(close_1d, np.nan)
    direction = np.full_like(close_1d, np.nan)  # 1 for uptrend, -1 for downtrend
    
    # Initialize first valid Supertrend value
    for i in range(atr_period, len(close_1d)):
        if i == atr_period:
            supertrend[i] = upper_band[i]
            direction[i] = 1  # start with uptrend assumption
        else:
            # Supertrend logic
            if close_1d[i-1] > supertrend[i-1]:
                supertrend[i] = max(lower_band[i], supertrend[i-1])
                direction[i] = 1
            else:
                supertrend[i] = min(upper_band[i], supertrend[i-1])
                direction[i] = -1
            
            # Band adjustments
            if supertrend[i] < supertrend[i-1] and close_1d[i-1] > supertrend[i-1]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            if supertrend[i] > supertrend[i-1] and close_1d[i-1] < supertrend[i-1]:
                supertrend[i] = lower_band[i]
                direction[i] = 1
    
    # Align Supertrend direction to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, direction)
    
    # Calculate Camarilla levels from prior 4h bar (using prior bar's HLC)
    prior_high = np.concatenate([[np.nan], high[:-1]])
    prior_low = np.concatenate([[np.nan], low[:-1]])
    prior_close = np.concatenate([[np.nan], close[:-1]])
    
    hl_range = prior_high - prior_low
    camarilla_r3 = prior_close + hl_range * 1.1 / 4
    camarilla_s3 = prior_close - hl_range * 1.1 / 4
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(atr_period + 5, 20)  # Need Supertrend and volume MA20
    
    for i in range(start_idx, n):
        if (np.isnan(supertrend_aligned[i]) or np.isnan(prior_high[i]) or np.isnan(prior_low[i]) or 
            np.isnan(prior_close[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: Supertrend direction (1=uptrend, -1=downtrend)
        uptrend = supertrend_aligned[i] == 1
        downtrend = supertrend_aligned[i] == -1
        
        # Breakout conditions
        breakout_long = close[i] > camarilla_r3[i]  # Price breaks above R3
        breakout_short = close[i] < camarilla_s3[i]  # Price breaks below S3
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R3 with volume spike and uptrend on 1d
            if breakout_long and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below S3 with volume spike and downtrend on 1d
            elif breakout_short and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on close below prior bar's low or trend reversal
            if close[i] < prior_low[i] or supertrend_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on close above prior bar's high or trend reversal
            if close[i] > prior_high[i] or supertrend_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals