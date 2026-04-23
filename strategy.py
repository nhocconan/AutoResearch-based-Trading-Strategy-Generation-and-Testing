#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R3 level and close > 4h EMA34 (uptrend) with volume > 1.5x average.
Short when price breaks below Camarilla S3 level and close < 4h EMA34 (downtrend) with volume > 1.5x average.
Exit on opposite Camarilla level break or trend reversal. Uses 1h timeframe targeting 60-150 total trades over 4 years.
Camarilla levels provide precise intraday support/resistance, EMA34 filters medium-term trend, volume spike confirms breakout strength.
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
    open_time = prices['open_time'].values
    
    # Load 4h data for EMA34 trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 4h EMA34 to 1h timeframe
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (already datetime64[ms], use DatetimeIndex)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(vol_ma[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_val = ema34_4h_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        # Calculate Camarilla levels for today using previous day's OHLC from 1d data
        # Load 1d data ONCE per loop would be inefficient, so we approximate using 24-period lookback on 4h data (6*4h=24h)
        # This is acceptable as we only need the previous day's OHLC for Camarilla calculation
        if i >= 6:  # Need at least 6*4h bars = 24h lookback, but we're on 1h timeframe so 24*4 = 96 bars
            lookback_start = i - 96  # 24 * 4h bars = 96 1h bars
            if lookback_start >= 0:
                # Get the 4h bars for the previous day (24 bars back from current 4h bar)
                # We need to map 1h index to 4h index for the lookback
                # For simplicity, we'll use rolling window on 1h data with 24*4=96 period lookback
                # This gives us the previous day's high/low/close approximation
                lookback_start_1h = i - 96
                if lookback_start_1h >= 0:
                    prev_high = np.max(high[lookback_start_1h:i])
                    prev_low = np.min(low[lookback_start_1h:i])
                    prev_close = close[i-1]  # previous bar close
                    
                    # Camarilla levels
                    range_val = prev_high - prev_low
                    camarilla_r3 = prev_close + (range_val * 1.1 / 4)
                    camarilla_s3 = prev_close - (range_val * 1.1 / 4)
                else:
                    # Not enough data for Camarilla calculation
                    if position != 0:
                        signals[i] = 0.0
                        position = 0
                    continue
            else:
                # Not enough data for Camarilla calculation
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                continue
        else:
            # Not enough data for Camarilla calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND price > 4h EMA34 (uptrend) AND volume spike
            if (price > camarilla_r3 and price > ema34_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: price breaks below Camarilla S3 AND price < 4h EMA34 (downtrend) AND volume spike
            elif (price < camarilla_s3 and price < ema34_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.20
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
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R3_S3_4hEMA34_VolumeSpike"
timeframe = "1h"
leverage = 1.0