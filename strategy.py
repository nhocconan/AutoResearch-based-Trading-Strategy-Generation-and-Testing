#!/usr/bin/env python3
# 4h_12h_camarilla_volume_rsi_breakout_v1
# Strategy: 4-hour Camarilla pivot breakout from 12-hour levels with volume confirmation and RSI filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Combines 12-hour Camarilla levels (wider bands for fewer, stronger signals) with
# volume spikes and RSI(14) > 50 for longs / < 50 for shorts to filter counter-trend noise.
# Targets 20-40 trades per year by using higher timeframe pivot levels and multiple confirmations.
# Works in bull markets via breakout logic and in bear markets via rejection logic at R3/S3 levels.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_volume_rsi_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h OHLC for Camarilla pivots
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels for previous 12h bar
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    r4_12h = close_12h + range_12h * 1.1 / 2.0
    r3_12h = close_12h + range_12h * 1.1 / 4.0
    s3_12h = close_12h - range_12h * 1.1 / 4.0
    s4_12h = close_12h - range_12h * 1.1 / 2.0
    
    # RSI(14) for momentum filter
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align 12h data to 4h timeframe (wait for 12h bar close)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi_values)
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_12h_aligned[i]) or np.isnan(r3_12h_aligned[i]) or 
            np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(rsi_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Momentum filter: RSI > 50 for bullish bias, < 50 for bearish bias
        bullish_momentum = rsi_aligned[i] > 50
        bearish_momentum = rsi_aligned[i] < 50
        
        # Camarilla signals (using previous 12h bar's levels)
        breakout_up = price_close > r4_12h_aligned[i]   # Break above R4
        breakdown_down = price_close < s4_12h_aligned[i]  # Break below S4
        reject_at_r3 = price_close < r3_12h_aligned[i] and price_close > r3_12h_aligned[i-1]  # Reject at R3
        reject_at_s3 = price_close > s3_12h_aligned[i] and price_close < s3_12h_aligned[i-1]  # Reject at S3
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: Break above R4 with volume in bullish momentum OR rejection at R3 with volume in bullish momentum
        long_signal = (breakout_up and vol_confirmed and bullish_momentum) or \
                      (reject_at_r3 and vol_confirmed and bullish_momentum)
        
        # Short: Break below S4 with volume in bearish momentum OR rejection at S3 with volume in bearish momentum
        short_signal = (breakdown_down and vol_confirmed and bearish_momentum) or \
                       (reject_at_s3 and vol_confirmed and bearish_momentum)
        
        # Exit when price returns to the 12h pivot level or opposite Camarilla level
        pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
        exit_long = position == 1 and (price_close < pivot_12h_aligned[i] or 
                                       price_close < s3_12h_aligned[i])
        exit_short = position == -1 and (price_close > pivot_12h_aligned[i] or 
                                         price_close > r3_12h_aligned[i])
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals