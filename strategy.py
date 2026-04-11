#!/usr/bin/env python3
# 1d_1w_camarilla_volume_v1
# Strategy: 1d Camarilla pivot mean-reversion with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: In ranging markets, price reverts to Camarilla pivot levels (H3/L3). 
# Weekly trend filter avoids counter-trend trades. Volume confirms genuine interest.
# Designed for low frequency (10-25 trades/year) to minimize fee drift in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily ATR(14) for volatility
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.insert(tr, 0, 0)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    
    # Daily Camarilla levels (based on previous day)
    # H3 = close + 1.1*(high - low)/2
    # L3 = close - 1.1*(high - low)/2
    # H4 = close + 1.5*(high - low)/2
    # L4 = close - 1.5*(high - low)/2
    # Using previous day's range
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    camarilla_h4 = prev_close + 1.5 * (prev_high - prev_low) / 2
    camarilla_l4 = prev_close - 1.5 * (prev_high - prev_low) / 2
    
    # Volume filter: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr.iloc[i]) or 
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Weekly trend filter: only trade in direction of weekly trend
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Mean reversion at Camarilla levels with volume confirmation
        long_signal = (close[i] <= camarilla_l3[i] and 
                      close[i] > camarilla_l4[i] and 
                      weekly_uptrend and 
                      volume_filter[i])
        
        short_signal = (close[i] >= camarilla_h3[i] and 
                       close[i] < camarilla_h4[i] and 
                       weekly_downtrend and 
                       volume_filter[i])
        
        # Exit when price moves back to pivot or opposite signal
        exit_long = (position == 1 and 
                    (close[i] >= (camarilla_h3[i] + camarilla_l3[i])/2 or  # midpoint
                     not weekly_uptrend))
        
        exit_short = (position == -1 and 
                     (close[i] <= (camarilla_h3[i] + camarilla_l3[i])/2 or
                      not weekly_downtrend))
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals