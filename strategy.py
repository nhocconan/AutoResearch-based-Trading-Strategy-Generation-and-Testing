#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and 1w EMA34 trend filter.
# Uses ATR-based trailing stop for risk management. Discrete sizing 0.25.
# Target: 50-150 total trades over 4 years (12-37/year).
# Camarilla pivot levels provide high-probability reversal/breakout zones.
# 1d volume confirmation reduces false breakouts.
# 1w EMA34 ensures we only trade in the direction of the higher timeframe trend.
# Works in both bull and bear markets by following the weekly trend.

name = "12h_Camarilla_R3_S3_1dVolumeSpike_1wEMA34_ATRStop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels (R3, S3) from prior completed 1d bar
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 1 prior completed 1d bar
        return np.zeros(n)
    
    # Calculate prior 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    camarilla_high_1d = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_low_1d = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Shift by 1 to use only prior completed 1d bar (avoid look-ahead)
    prior_camarilla_high = np.roll(camarilla_high_1d, 1)
    prior_camarilla_high[0] = np.nan
    prior_camarilla_low = np.roll(camarilla_low_1d, 1)
    prior_camarilla_low[0] = np.nan
    
    # Align Camarilla levels to 12h timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, prior_camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, prior_camarilla_low)
    
    # Calculate 1d volume confirmation: volume > 2.0x 24-bar average (on 1d data)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=24, min_periods=24).mean().values
    volume_spike_1d = vol_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate 1w EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:  # Need at least 34 periods for EMA + 1 for prior
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR(24) for stoploss (using 12h data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=24, min_periods=24, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0
    lowest_low_since_entry = 0
    
    for i in range(100, n):  # Start after sufficient warmup
        # Get current values
        camarilla_high_val = camarilla_high_aligned[i]
        camarilla_low_val = camarilla_low_aligned[i]
        vol_spike = volume_spike_1d_aligned[i]
        ema_trend = ema_34_1w_aligned[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if np.isnan(camarilla_high_val) or np.isnan(camarilla_low_val) or np.isnan(ema_trend) or np.isnan(atr_val):
            continue
            
        # Entry conditions
        # Long: break above prior 1d Camarilla R3 with volume spike and above 1w EMA34
        long_entry = (close[i] > camarilla_high_val) and vol_spike and (close[i] > ema_trend)
        # Short: break below prior 1d Camarilla S3 with volume spike and below 1w EMA34
        short_entry = (close[i] < camarilla_low_val) and vol_spike and (close[i] < ema_trend)
        
        # Exit conditions (trailing stop)
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
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            elif short_entry:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals