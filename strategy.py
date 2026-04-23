#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R3 level and close > 12h EMA34 (uptrend) with volume > 1.8x average.
Short when price breaks below Camarilla S3 level and close < 12h EMA34 (downtrend) with volume > 1.8x average.
Exit on opposite Camarilla level break or trend reversal. Uses 4h timeframe targeting 75-200 total trades over 4 years.
Camarilla R3/S3 levels provide stronger intraday support/resistance than R1/S1, reducing false breakouts.
EMA34 filters medium-term trend, volume spike confirms breakout strength. Designed to capture strong momentum moves
while avoiding whipsaws in choppy markets across both bull and bear regimes. Tighter volume threshold and stronger
Camarilla levels aim to reduce trade frequency vs previous variants while maintaining edge.
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
    
    # Load 12h data for EMA34 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA34 to 4h timeframe
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_val = ema34_12h_aligned[i]
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
            camarilla_r3 = prev_close + (range_val * 1.1 / 4)  # R3 level
            camarilla_s3 = prev_close - (range_val * 1.1 / 4)  # S3 level
        else:
            # Not enough data for Camarilla calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND price > 12h EMA34 (uptrend) AND volume spike
            if (price > camarilla_r3 and price > ema34_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Camarilla S3 AND price < 12h EMA34 (downtrend) AND volume spike
            elif (price < camarilla_s3 and price < ema34_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Camarilla S3 OR trend reversal
                if (price < camarilla_s3 or price < ema34_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Camarilla R3 OR trend reversal
                if (price > camarilla_r3 or price > ema34_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3_S3_12hEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0