#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power + 12h EMA Trend Filter with Volume Spike Confirmation
# Uses 12h EMA50 for trend direction (avoid counter-trend trades) and Elder Ray to measure
# bull/bear power behind price moves. Volume spike confirms institutional participation.
# Long when: EMA50 up AND Bull Power > 0 AND Volume > 1.5 * Volume MA20
# Short when: EMA50 down AND Bear Power < 0 AND Volume > 1.5 * Volume MA20
# Uses discrete sizing 0.25. Target: 15-30 trades/year (60-120 over 4 years).
# Works in bull markets (trend following with power) and bear markets (avoids false signals
# via trend filter, only takes power-aligned entries during retracements).

name = "6h_ElderRay_Power_12hEMA50_VolumeSpike_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 12h data ONCE before loop for EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h EMA50 slope (trend direction)
    ema_slope = np.zeros_like(ema_50_12h_aligned)
    ema_slope[1:] = ema_50_12h_aligned[1:] - ema_50_12h_aligned[:-1]
    
    # Elder Ray Power components
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: Volume > 1.5 * 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_slope[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema_slope = ema_slope[i]
        curr_volume_spike = volume_spike[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Uptrend AND Bull Power > 0 AND Volume Spike
            if (curr_ema_slope > 0 and 
                curr_bull_power > 0 and 
                curr_volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Downtrend AND Bear Power < 0 AND Volume Spike
            elif (curr_ema_slope < 0 and 
                  curr_bear_power < 0 and 
                  curr_volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Trend turns down OR Bull Power <= 0
            if (curr_ema_slope <= 0 or 
                curr_bull_power <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Trend turns up OR Bear Power >= 0
            if (curr_ema_slope >= 0 or 
                curr_bear_power >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals