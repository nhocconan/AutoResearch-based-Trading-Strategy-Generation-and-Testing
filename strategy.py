#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume spike
# Camarilla pivot levels identify key intraday support/resistance. Break of R3/S3 levels
# with 12h EMA34 trend alignment and volume spike captures strong momentum moves.
# Uses discrete sizing 0.25 to minimize fee churn. Target: 75-200 trades over 4 years.
# Works in bull markets (breakouts continue trend) and bear markets (breakdowns continue downtrend).

name = "4h_Camarilla_R3S3_12hEMA34_Volume_v1"
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
    
    # Calculate 12h OHLC for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Prior 12h OHLC for Camarilla calculation (shift to avoid look-ahead)
    prior_high = df_12h['high'].shift(1).values
    prior_low = df_12h['low'].shift(1).values
    prior_close = df_12h['close'].shift(1).values
    
    # Calculate Camarilla pivot levels for prior 12h bar
    # Pivot = (H + L + C) / 3
    pivot = (prior_high + prior_low + prior_close) / 3
    # Range = H - L
    rng = prior_high - prior_low
    # Resistance levels: R3 = pivot + (H-L)*1.1/2, R4 = pivot + (H-L)*1.1
    # Support levels: S3 = pivot - (H-L)*1.1/2, S4 = pivot - (H-L)*1.1
    camarilla_r3 = pivot + (rng * 1.1 / 2)
    camarilla_s3 = pivot - (rng * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe (wait for 12h bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # 12h EMA34 for trend filter
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20, 34, 50)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_ema_34_12h = ema_34_12h_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Camarilla break and 12h trend filter
            if curr_volume_spike:
                # Bullish: Close breaks above R3 + price above 12h EMA34
                if curr_close > curr_r3 and curr_close > curr_ema_34_12h:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Close breaks below S3 + price below 12h EMA34
                elif curr_close < curr_s3 and curr_close < curr_ema_34_12h:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: Close drops below pivot or loses 12h trend
            pivot_level = (curr_r3 + curr_s3) / 2  # approximate pivot as midpoint
            if curr_close < pivot_level or curr_close < curr_ema_34_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close rises above pivot or loses 12h trend
            pivot_level = (curr_r3 + curr_s3) / 2  # approximate pivot as midpoint
            if curr_close > pivot_level or curr_close > curr_ema_34_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals