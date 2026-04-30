#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation
# Camarilla pivots identify key intraday support/resistance levels. Break of R3/S3 levels with
# 1w EMA34 trend alignment and volume spike captures strong momentum moves in both bull and bear markets.
# Uses discrete sizing 0.25 to minimize fee churn. Target: 50-150 trades over 4 years (12-37/year).
# Works in bull markets (breakouts continue trend) and bear markets (breakdowns continue downtrend).

name = "12h_Camarilla_R3S3_Breakout_1wEMA34_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w Camarilla pivot levels from prior 1w bar
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Prior 1w OHLC for Camarilla calculation (shift to avoid look-ahead)
    prior_high = df_1w['high'].shift(1).values
    prior_low = df_1w['low'].shift(1).values
    prior_close = df_1w['close'].shift(1).values
    
    # Calculate Camarilla levels on 1w timeframe
    # R3 = Close + (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/4
    camarilla_r3 = prior_close + (prior_high - prior_low) * 1.1 / 4
    camarilla_s3 = prior_close - (prior_high - prior_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (wait for 1w bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20, 34)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Camarilla break and 1w trend filter
            if curr_volume_spike:
                # Bullish: Close breaks above R3 + price above 1w EMA34
                if curr_close > curr_r3 and curr_close > curr_ema_34_1w:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Close breaks below S3 + price below 1w EMA34
                elif curr_close < curr_s3 and curr_close < curr_ema_34_1w:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: Close drops below Camarilla midpoint or loses 1w trend
            camarilla_mid = (curr_r3 + curr_s3) / 2
            if curr_close < camarilla_mid or curr_close < curr_ema_34_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close rises above Camarilla midpoint or loses 1w trend
            camarilla_mid = (curr_r3 + curr_s3) / 2
            if curr_close > camarilla_mid or curr_close > curr_ema_34_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals