#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume spike confirmation
# Camarilla pivot levels from prior 4h candle identify intraday support/resistance; breakouts above R3 or below S3
# with volume confirmation indicate strong momentum. 4h EMA34 ensures trades align with higher timeframe trend
# to avoid false breakouts in choppy markets. Session filter (08-20 UTC) reduces noise. Designed for 60-150 total trades
# over 4 years (15-37/year) on 1h timeframe. Uses 4h for signal direction, 1h only for entry timing.

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA34 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate prior 4h candle's Camarilla levels (using 4h data)
    # Camarilla: based on prior 4h candle's high, low, close
    prior_high_4h = df_4h['high'].shift(1).values  # prior 4h candle's high
    prior_low_4h = df_4h['low'].shift(1).values    # prior 4h candle's low
    prior_close_4h = df_4h['close'].shift(1).values # prior 4h candle's close
    
    # Calculate Camarilla levels (R3/S3 are significant breakout levels)
    R3 = prior_close_4h + (prior_high_4h - prior_low_4h) * 1.1 / 4
    S3 = prior_close_4h - (prior_high_4h - prior_low_4h) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (wait for prior 4h candle to complete)
    R3_aligned = align_htf_to_ltf(prices, df_4h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_4h, S3)
    
    # Volume confirmation: 2.0x 20-period average (~10h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA34 and prior 4h data)
    start_idx = max(35, 30)  # 35 bars for EMA34, 30 bars to ensure prior 4h data available
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Check for NaN values in indicators
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R3 with volume spike AND price > 4h EMA34 (bullish trend)
            if (close[i] > R3_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S3 with volume spike AND price < 4h EMA34 (bearish trend)
            elif (close[i] < S3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below R3 (failed breakout) OR price below 4h EMA34 (trend change)
            if close[i] < R3_aligned[i] or close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price crosses above S3 (failed breakdown) OR price above 4h EMA34 (trend change)
            if close[i] > S3_aligned[i] or close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals