#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R4/S4 breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above Camarilla R4 level and close > 1w EMA50 (uptrend) with volume > 1.8x average.
Short when price breaks below Camarilla S4 level and close < 1w EMA50 (downtrend) with volume > 1.8x average.
Exit on opposite Camarilla level break or trend reversal. Uses 12h timeframe targeting 50-150 total trades over 4 years.
Camarilla R4/S4 levels provide stronger support/resistance than R3/S3, reducing false breakouts.
1w EMA50 filters long-term trend, volume confirmation ensures breakout strength.
Designed to capture strong momentum moves while avoiding whipsaws in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 12h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_1w_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        # Calculate Camarilla levels for today using previous day's OHLC
        # Need to get previous day's high, low, close from 1d data
        # We'll approximate using rolling window on 12h data for simplicity
        # In practice, we'd use actual daily OHLC, but for now use 2-period lookback (12h * 2 = 24h)
        if i >= 2:
            lookback_start = i - 2
            prev_high = np.max(high[lookback_start:i])
            prev_low = np.min(low[lookback_start:i])
            prev_close = close[i-1]  # previous bar close
            
            # Camarilla levels
            range_val = prev_high - prev_low
            camarilla_r4 = prev_close + (range_val * 1.1 / 2)
            camarilla_s4 = prev_close - (range_val * 1.1 / 2)
        else:
            # Not enough data for Camarilla calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R4 AND price > 1w EMA50 (uptrend) AND volume spike
            if (price > camarilla_r4 and price > ema50_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Camarilla S4 AND price < 1w EMA50 (downtrend) AND volume spike
            elif (price < camarilla_s4 and price < ema50_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Camarilla S4 OR trend reversal
                if (price < camarilla_s4 or price < ema50_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Camarilla R4 OR trend reversal
                if (price > camarilla_r4 or price > ema50_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R4_S4_1wEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0