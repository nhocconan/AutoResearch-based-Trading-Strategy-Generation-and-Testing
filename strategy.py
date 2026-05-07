#!/usr/bin/env python3
name = "6h_Adaptive_Kelly_CamR3S3_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1d data for Camarilla levels (previous day's high, low, close)
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Calculate Camarilla R3 and S3 levels from previous 1d bar
    range_hl = prev_high - prev_low
    r3 = prev_close + range_hl * 0.55  # Standard Camarilla formula: close + (high-low)*1.1/2
    s3 = prev_close - range_hl * 0.55  # Standard Camarilla formula: close - (high-low)*1.1/2
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: 20-period average volume for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility filter: avoid low volatility (ATR > 0.4% of price)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    vol_filter = atr > 0.004 * close  # ATR > 0.4% of price
    
    # Session filter: 08:00 - 20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Kelly sizing components
    # Win probability estimate from recent price action
    returns_6h = np.diff(np.log(close), prepend=np.log(close[0]))
    win_prob = pd.Series(returns_6h > 0).rolling(window=50, min_periods=20).mean().values
    # Win/loss ratio from recent volatility
    abs_returns = np.abs(returns_6h)
    win_loss_ratio = pd.Series(abs_returns).rolling(window=50, min_periods=20).mean().values / \
                     (pd.Series(abs_returns[returns_6h < 0]).rolling(window=50, min_periods=20).mean().values + 1e-10)
    # Kelly fraction: f* = (bp - q)/b where b = win_loss_ratio, p = win_prob, q = 1-p
    kelly_raw = (win_loss_ratio * win_prob - (1 - win_prob)) / (win_loss_ratio + 1e-10)
    kelly_fraction = np.clip(kelly_raw, 0, 0.5)  # Cap at 50% Kelly
    position_size = kelly_fraction * 0.5  # Scale to 0-0.25 range
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 50)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN or invalid
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(vol_filter[i]) or not vol_filter[i] or
            not session_filter[i] or
            np.isnan(position_size[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period average
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above R3, above 1d EMA34 (uptrend), with volume spike
            buffer = 0.001 * close[i]  # 0.1% buffer to avoid whipsaws
            if (close[i] > r3_aligned[i] + buffer and 
                close[i] > ema_34_1d_aligned[i] + buffer and   # 1d uptrend
                volume_spike):
                signals[i] = position_size[i]
                position = 1
            # Short: Price breaks below S3, below 1d EMA34 (downtrend), with volume spike
            elif (close[i] < s3_aligned[i] - buffer and 
                  close[i] < ema_34_1d_aligned[i] - buffer and   # 1d downtrend
                  volume_spike):
                signals[i] = -position_size[i]
                position = -1
        elif position != 0:
            # Exit: Price returns to midpoint of prior 1d range (H3/L3)
            range_hl = prev_high - prev_low
            h3 = prev_close + range_hl * 0.275  # Standard Camarilla H3: close + (high-low)*1.1/4
            l3 = prev_close - range_hl * 0.275  # Standard Camarilla L3: close - (high-low)*1.1/4
            h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
            l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
            
            camarilla_mid = (h3_aligned[i] + l3_aligned[i]) / 2
            range_hl_1d = h3_aligned[i] - l3_aligned[i]
            # Exit when within 30% of midpoint (tighter exit to reduce holding losing positions)
            at_mid = abs(close[i] - camarilla_mid) < range_hl_1d * 0.30
            
            if at_mid:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position with Kelly sizing
                signals[i] = position_size[i] if position == 1 else -position_size[i]
    
    return signals