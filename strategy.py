#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivot levels provide high-probability intraday support/resistance in ranging markets.
# 1d EMA34 ensures we trade with the higher timeframe trend, adapting to bull/bear regimes.
# Volume confirmation filters false breakouts. Target: 75-200 total trades over 4 years (19-50/year).
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Need to get daily OHLC for each 4h bar's reference point
    camarilla_R3 = np.full(n, np.nan)
    camarilla_S3 = np.full(n, np.nan)
    camarilla_R4 = np.full(n, np.nan)
    camarilla_S4 = np.full(n, np.nan)
    
    # Resample to daily using actual Binance daily data alignment
    # We'll use the 1d data we already loaded
    for i in range(n):
        # Find the most recent completed 1d bar
        dt = open_time.iloc[i]
        # Convert to daily boundary (00:00 UTC)
        daily_dt = pd.Timestamp(dt.year, dt.month, dt.day)
        # Find index in 1d data for this date
        try:
            # Get index of the daily bar that completed before current time
            idx_1d = df_1d.index.get_indexer([daily_dt], method='ffill')[0]
            if idx_1d > 0:  # Ensure we have a previous day
                idx_1d_prev = idx_1d - 1
                if idx_1d_prev >= 0 and idx_1d_prev < len(df_1d):
                    high_1d = df_1d['high'].iloc[idx_1d_prev]
                    low_1d = df_1d['low'].iloc[idx_1d_prev]
                    close_1d = df_1d['close'].iloc[idx_1d_prev]
                    range_1d = high_1d - low_1d
                    camarilla_R3[i] = close_1d + range_1d * 1.1 / 4
                    camarilla_S3[i] = close_1d - range_1d * 1.1 / 4
                    camarilla_R4[i] = close_1d + range_1d * 1.1 / 2
                    camarilla_S4[i] = close_1d - range_1d * 1.1 / 2
        except:
            pass
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_R3[i]) or np.isnan(camarilla_S3[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.3 x 20-period EMA
        volume_confirm = volume[i] > (1.3 * vol_ema_20[i])
        
        if position == 0:
            # Long breakout: price closes above Camarilla R3 + volume + 1d EMA34 uptrend
            if (close[i] > camarilla_R3[i-1] and 
                volume_confirm and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: price closes below Camarilla S3 + volume + 1d EMA34 downtrend
            elif (close[i] < camarilla_S3[i-1] and 
                  volume_confirm and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla S3 OR 1d EMA34 turns down
            if (close[i] < camarilla_S3[i-1] or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Camarilla R3 OR 1d EMA34 turns up
            if (close[i] > camarilla_R3[i-1] or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals