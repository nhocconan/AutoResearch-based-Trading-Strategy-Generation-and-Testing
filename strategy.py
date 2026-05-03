#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation.
# Uses 1h timeframe for entry timing, 4h for HTF direction and pivot calculation.
# Breakouts above R1 (long) or below S1 (short) with volume confirmation and trend alignment.
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
# Session filter (08-20 UTC) reduces noise trades. Discrete sizing 0.20.

name = "1h_Camarilla_R1_S1_4hEMA50_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute hour for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data for Camarilla calculation and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Use prior completed 4h bar's OHLC for Camarilla calculation
    prior_close = np.roll(df_4h['close'].values, 1)
    prior_high = np.roll(df_4h['high'].values, 1)
    prior_low = np.roll(df_4h['low'].values, 1)
    prior_close[0] = np.nan
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    
    # Calculate Camarilla levels for prior 4h bar
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = prior_close + (prior_high - prior_low) * 1.1 / 12
    camarilla_s1 = prior_close - (prior_high - prior_low) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Calculate 4h EMA50 trend filter
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Calculate 4h volume regime (high volume when current volume > 1.5x 20-period MA)
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_regime = vol_4h > (1.5 * vol_ma_4h)  # High volume regime
    
    # Align volume regime to 1h timeframe
    vol_regime_aligned = align_htf_to_ltf(prices, df_4h, vol_regime)
    
    # Calculate ATR(14) for 1h data (for stoploss)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0
    lowest_low_since_entry = 0
    
    for i in range(100, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Skip if outside session
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Get current values
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        ema_trend = ema_50_aligned[i]
        vol_reg = vol_regime_aligned[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if np.isnan(r1) or np.isnan(s1) or np.isnan(ema_trend) or np.isnan(vol_reg) or np.isnan(atr_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Volume confirmation: current 1h volume > 1.5x 20-period MA
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        volume_spike = volume[i] > (1.5 * vol_ma_20)
        
        # Entry conditions
        # Long: break above R1 with volume spike, above 4h EMA50, and in high volume regime
        long_entry = (close[i] > r1) and volume_spike and (close[i] > ema_trend) and vol_reg
        # Short: break below S1 with volume spike, below 4h EMA50, and in high volume regime
        short_entry = (close[i] < s1) and volume_spike and (close[i] < ema_trend) and vol_reg
        
        # Exit conditions (ATR-based trailing stop)
        long_exit = False
        short_exit = False
        
        if position == 1:  # Long position
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            long_exit = close[i] < (highest_high_since_entry - 2.5 * atr_val)
        elif position == -1:  # Short position
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            short_exit = close[i] > (lowest_low_since_entry + 2.5 * atr_val)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.20
                position = 1
                highest_high_since_entry = high[i]
            elif short_entry:
                signals[i] = -0.20
                position = -1
                lowest_low_since_entry = low[i]
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals