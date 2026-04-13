#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d Bollinger Bands squeeze + RSI reversal + volume confirmation.
# Long: Bollinger Bands width < 20th percentile (squeeze) + RSI < 30 + volume > 1.3x avg volume.
# Short: Bollinger Bands width < 20th percentile (squeeze) + RSI > 70 + volume > 1.3x avg volume.
# Uses Bollinger squeeze to identify low volatility periods primed for breakout/reversal,
# RSI for overbought/oversold conditions, volume to confirm participation.
# Position size: 0.25 (25%). Target: 20-50 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # 1d data for Bollinger Bands and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Bollinger Bands (20, 2.0)
    bb_period = 20
    bb_mult = 2.0
    sma_20 = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = sma_20 + bb_mult * std_20
    bb_lower = sma_20 - bb_mult * std_20
    bb_width = bb_upper - bb_lower
    
    # Percentile rank of BB width (252-period lookback ~ 1 year)
    bb_width_pct = np.full(len(bb_width), np.nan)
    for i in range(252, len(bb_width)):
        bb_width_pct[i] = (bb_width[i] <= bb_width[i-252:i]).sum() / 252 * 100
    
    # RSI (14)
    rsi_period = 14
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Average volume (20-period) for volume confirmation
    avg_volume_1d = np.full(len(volume), np.nan)
    for i in range(20, len(volume)):
        avg_volume_1d[i] = np.mean(volume[i-20:i])
    
    # Align 1d indicators to 4h
    bb_width_pct_aligned = align_htf_to_ltf(prices, df_1d, bb_width_pct)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    avg_volume_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(252, n):  # Start after BB percentile lookback
        # Skip if any required data is not ready
        if (np.isnan(bb_width_pct_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(avg_volume_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if not (8 <= hour <= 20):
            signals[i] = 0.0
            continue
        
        bb_width_pct_val = bb_width_pct_aligned[i]
        rsi_val = rsi_aligned[i]
        vol = volume[i]
        avg_vol = avg_volume_aligned[i]
        
        # Bollinger squeeze: width < 20th percentile
        squeeze = bb_width_pct_val < 20
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long: squeeze + RSI < 30 (oversold) + volume confirmation
            if squeeze and (rsi_val < 30) and volume_confirm:
                position = 1
                signals[i] = position_size
            # Short: squeeze + RSI > 70 (overbought) + volume confirmation
            elif squeeze and (rsi_val > 70) and volume_confirm:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI > 50 (exit oversold condition) or squeeze breaks
            if (rsi_val > 50) or (bb_width_pct_val >= 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI < 50 (exit overbought condition) or squeeze breaks
            if (rsi_val < 50) or (bb_width_pct_val >= 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Bollinger_Squeeze_RSI_Volume"
timeframe = "4h"
leverage = 1.0