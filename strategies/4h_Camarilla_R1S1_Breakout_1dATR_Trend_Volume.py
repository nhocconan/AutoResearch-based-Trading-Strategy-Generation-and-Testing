#!/usr/bin/env python3

name = "4h_Camarilla_R1S1_Breakout_1dATR_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR, trend filter, and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.nan], tr1])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d_prev = high_1d[:-1]
    low_1d_prev = low_1d[:-1]
    close_1d_prev = close_1d[:-1]
    high_1d_prev = np.concatenate([[np.nan], high_1d_prev])
    low_1d_prev = np.concatenate([[np.nan], low_1d_prev])
    close_1d_prev = np.concatenate([[np.nan], close_1d_prev])
    
    # R1 = L1 = close + 1.1*(high-low)/12
    # S1 = H1 = close - 1.1*(high-low)/12
    high_low = high_1d_prev - low_1d_prev
    r1 = close_1d_prev + 1.1 * high_low / 12
    s1 = close_1d_prev - 1.1 * high_low / 12
    
    # Align 1d indicators to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Volume filter: current volume > 1.5x 20-period average (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 6  # ~1 day for 4h to reduce trades
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 1d trend direction
        trend_1d_up = close_1d_aligned[i] > ema_34_1d_aligned[i]
        trend_1d_down = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: break above R1 with volume in 1d uptrend and sufficient volatility
            if (close[i] > r1_aligned[i] and 
                trend_1d_up and 
                vol_filter[i] and
                atr_14_1d_aligned[i] > 0):  # Ensure volatility is present
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: break below S1 with volume in 1d downtrend and sufficient volatility
            elif (close[i] < s1_aligned[i] and 
                  trend_1d_down and 
                  vol_filter[i] and
                  atr_14_1d_aligned[i] > 0):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: close back below S1 or trend change
            if (close[i] < s1_aligned[i]) or not trend_1d_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: close back above R1 or trend change
            if (close[i] > r1_aligned[i]) or not trend_1d_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R1/S1 breakout with 1d EMA34 trend filter, volume confirmation, and ATR volatility filter on 4h timeframe.
# Long when price breaks above R1 with volume spike in 1d uptrend and sufficient volatility.
# Short when price breaks below S1 with volume spike in 1d downtrend and sufficient volatility.
# Exits when price returns to S1/R1 or trend changes.
# Uses ATR(14) to avoid trading in low volatility periods.
# Volume confirmation avoids false breakouts. Cooldown (1 day) reduces trade frequency.
# Target: 20-40 trades/year. Works in bull markets by catching breakouts in uptrends
# and in bear markets by shorting breakdowns in downtrends. 4h timeframe balances 
# signal quality and trade frequency. Camarilla levels provide precise intraday
# support/resistance derived from prior day's range.