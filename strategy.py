#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot (R1/S1) breakout with 1d EMA trend filter and volume confirmation.
# Uses daily EMA34 for trend direction and Camarilla levels from previous day for entry.
# Designed for 20-40 trades/year to avoid fee drag. Works in bull/bear via trend filter.
# Target: ETH/BTC with potential for SOL.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla levels (R1, S1)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Use previous day's data to avoid look-ahead
    hl_range_1d = high_1d - low_1d
    camarilla_r1_1d = close_1d + hl_range_1d * 1.1 / 12
    camarilla_s1_1d = close_1d - hl_range_1d * 1.1 / 12
    
    # Shift to get previous day's levels (available at close of previous day)
    camarilla_r1_1d_prev = np.roll(camarilla_r1_1d, 1)
    camarilla_s1_1d_prev = np.roll(camarilla_s1_1d, 1)
    camarilla_r1_1d_prev[0] = np.nan  # first day has no previous
    camarilla_s1_1d_prev[0] = np.nan
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily indicators to 4h timeframe
    camarilla_r1_4h = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d_prev)
    camarilla_s1_4h = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d_prev)
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h ATR for stop loss and entry threshold
    tr_4h_1 = high - low
    tr_4h_2 = np.abs(high - np.roll(close, 1))
    tr_4h_3 = np.abs(low - np.roll(close, 1))
    tr_4h_1[0] = high[0] - low[0]
    tr_4h_2[0] = np.abs(high[0] - close[0])
    tr_4h_3[0] = np.abs(low[0] - close[0])
    tr_4h = np.maximum(tr_4h_1, np.maximum(tr_4h_2, tr_4h_3))
    atr_4h = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # need daily EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_4h[i]) or np.isnan(camarilla_s1_4h[i]) or 
            np.isnan(ema_34_4h[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above daily EMA34 (uptrend) or below (downtrend)
        trend_up = close[i] > ema_34_4h[i]
        trend_down = close[i] < ema_34_4h[i]
        
        if position == 0:
            # Long entry: price breaks above Camarilla R1 with volume and uptrend
            if (close[i] > camarilla_r1_4h[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Camarilla S1 with volume and downtrend
            elif (close[i] < camarilla_s1_4h[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below Camarilla S1 or ATR-based stop
            if close[i] < camarilla_s1_4h[i] - 0.5 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Camarilla R1 or ATR-based stop
            if close[i] > camarilla_r1_4h[i] + 0.5 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_EMA34_VolumeFilter"
timeframe = "4h"
leverage = 1.0