#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 level in 1d uptrend with volume > 1.5x 20-period average.
# Short when price breaks below Camarilla S3 level in 1d downtrend with volume > 1.5x 20-period average.
# ATR-based trailing stoploss (2.5*ATR) to manage risk. Uses discrete sizing 0.25.
# Camarilla levels provide intraday support/resistance structure proven effective in ranging and trending markets.
# 1d EMA34 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws.
# Volume confirmation filters breakouts with insufficient participation.
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels for 4h timeframe using previous bar's OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use previous bar to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    rang = prev_high - prev_low
    camarilla_r3 = prev_close + 1.1 * rang
    camarilla_s3 = prev_close - 1.1 * rang
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high = 0.0
    lowest_low = 0.0
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_ok = volume_spike[i]
        trend_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND 1d uptrend AND volume spike
            if close_val > camarilla_r3[i] and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_high = close_val
            # Short: price breaks below Camarilla S3 AND 1d downtrend AND volume spike
            elif close_val < camarilla_s3[i] and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_low = close_val
        elif position == 1:
            # Update highest high for trailing stop
            highest_high = max(highest_high, close_val)
            # ATR trailing stop: exit if price drops 2.5*ATR below highest high
            if close_val < highest_high - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low for trailing stop
            lowest_low = min(lowest_low, close_val)
            # ATR trailing stop: exit if price rises 2.5*ATR above lowest low
            if close_val > lowest_low + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals