#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1w EMA50 trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In ranging markets (2025+), 
# mean reversion at extremes works well. 1w EMA50 filter ensures trades align with 
# major trend to avoid counter-trend moves. Volume confirmation filters weak signals.
# Discrete sizing 0.25 to manage fees. ATR-based trailing stop (2.0x) for risk control.
# Target: 50-150 total trades over 4 years on BTC/ETH/SOL.

name = "6h_WilliamsR_MeanReversion_1wEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w OHLC for Williams %R and EMA (from prior completed 1w bar)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:  # Need at least 1 completed bar for prior
        return np.zeros(n)
    
    # Use prior completed 1w bar's OHLC for Williams %R calculation
    prior_high = np.roll(df_1w['high'].values, 1)
    prior_low = np.roll(df_1w['low'].values, 1)
    prior_close = np.roll(df_1w['close'].values, 1)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    # Calculate Williams %R(14) for prior 1w bar
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(prior_high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(prior_low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - prior_close) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid div by zero
    
    # Align Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    # Calculate 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(30) for stoploss (using 6h data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=30, min_periods=30, adjust=False).mean().values
    
    # Volume confirmation: volume > 2.0x 30-bar average (on 6h data)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0
    lowest_low_since_entry = 0
    
    for i in range(100, n):  # Start after sufficient warmup
        # Get current values
        wr = williams_r_aligned[i]
        ema_trend = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if np.isnan(wr) or np.isnan(ema_trend) or np.isnan(atr_val):
            continue
            
        # Entry conditions
        # Long: Williams %R < -80 (oversold) and price above 1w EMA50 and volume spike
        long_entry = (wr < -80) and (close[i] > ema_trend) and vol_spike
        # Short: Williams %R > -20 (overbought) and price below 1w EMA50 and volume spike
        short_entry = (wr > -20) and (close[i] < ema_trend) and vol_spike
        
        # Exit conditions (trailing stop)
        long_exit = False
        short_exit = False
        
        if position == 1:  # Long position
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            long_exit = close[i] < (highest_high_since_entry - 2.0 * atr_val)
        elif position == -1:  # Short position
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            short_exit = close[i] > (lowest_low_since_entry + 2.0 * atr_val)
        
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