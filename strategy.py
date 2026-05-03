#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation.
# Uses ATR-based trailing stop for risk management. Discrete sizing 0.20.
# Target: 60-150 total trades over 4 years (15-37/year).
# Uses 4h for signal direction (Camarilla breakouts) and 1d for trend filter (EMA50).
# 1h timeframe used only for entry timing precision and execution.
# Volume confirmation reduces false breakouts.
# Session filter (08-20 UTC) to reduce noise trades.

name = "1h_Camarilla_R3_S3_1dEMA50_VolumeSpike_ATRStop_v1"
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
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h Camarilla pivot levels (R3, S3) from prior completed 4h bar
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:  # Need at least 1 completed bar for prior
        return np.zeros(n)
    
    # Calculate typical price for Camarilla
    typical_price_4h = (df_4h['high'].values + df_4h['low'].values + df_4h['close'].values) / 3.0
    
    # Calculate prior completed 4h bar's high, low, close
    prior_high_4h = np.roll(df_4h['high'].values, 1)
    prior_low_4h = np.roll(df_4h['low'].values, 1)
    prior_close_4h = np.roll(df_4h['close'].values, 1)
    prior_high_4h[0] = np.nan
    prior_low_4h[0] = np.nan
    prior_close_4h[0] = np.nan
    
    # Calculate Camarilla R3 and S3 levels
    # R3 = prior_close + 1.1 * (prior_high - prior_low)
    # S3 = prior_close - 1.1 * (prior_high - prior_low)
    camarilla_r3_4h = prior_close_4h + 1.1 * (prior_high_4h - prior_low_4h)
    camarilla_s3_4h = prior_close_4h - 1.1 * (prior_high_4h - prior_low_4h)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    
    # Calculate 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss (using 1h data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Volume confirmation: volume > 2.0x 24-bar average (on 1h data)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0
    lowest_low_since_entry = 0
    
    for i in range(100, n):  # Start after sufficient warmup
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
            
        # Get current values
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if np.isnan(r3_val) or np.isnan(s3_val) or np.isnan(ema_trend) or np.isnan(atr_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
            
        # Entry conditions
        # Long: break above Camarilla R3 with volume spike and above 1d EMA50
        long_entry = (close[i] > r3_val) and (close[i] > ema_trend) and vol_spike
        # Short: break below Camarilla S3 with volume spike and below 1d EMA50
        short_entry = (close[i] < s3_val) and (close[i] < ema_trend) and vol_spike
        
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
                signals[i] = 0.20
                position = 1
                highest_high_since_entry = high[i]
            elif short_entry:
                signals[i] = -0.20
                position = -1
                lowest_low_since_entry = low[i]
            else:
                signals[i] = 0.0
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