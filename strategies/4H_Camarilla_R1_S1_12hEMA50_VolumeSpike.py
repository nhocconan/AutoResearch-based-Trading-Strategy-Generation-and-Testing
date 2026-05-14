#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R1 level and close > 12h EMA50 (uptrend) with volume > 2.0x average.
Short when price breaks below Camarilla S1 level and close < 12h EMA50 (downtrend) with volume > 2.0x average.
Exit on opposite Camarilla level break or trend reversal. Uses 4h timeframe targeting 75-200 total trades over 4 years.
Camarilla levels provide precise intraday support/resistance, EMA50 filters medium-term trend, volume spike confirms breakout strength.
Designed to capture strong momentum moves while avoiding whipsaws in choppy markets across both bull and bear regimes.
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
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_12h_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        # Calculate Camarilla levels for today using previous day's OHLC
        # Need to get previous day's high, low, close from 1d data
        # We'll approximate using rolling window on 4h data for simplicity
        # In practice, we'd use actual daily OHLC, but for now use 24-period lookback (4h * 6 = 24h)
        if i >= 24:
            lookback_start = i - 24
            prev_high = np.max(high[lookback_start:i])
            prev_low = np.min(low[lookback_start:i])
            prev_close = close[i-1]  # previous bar close
            
            # Camarilla levels
            range_val = prev_high - prev_low
            camarilla_r1 = prev_close + (range_val * 1.1 / 12)
            camarilla_s1 = prev_close - (range_val * 1.1 / 12)
        else:
            # Not enough data for Camarilla calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 AND price > 12h EMA50 (uptrend) AND volume spike
            if (price > camarilla_r1 and price > ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Camarilla S1 AND price < 12h EMA50 (downtrend) AND volume spike
            elif (price < camarilla_s1 and price < ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Camarilla S1 OR trend reversal
                if (price < camarilla_s1 or price < ema50_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Camarilla R1 OR trend reversal
                if (price > camarilla_r1 or price > ema50_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1_S1_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0