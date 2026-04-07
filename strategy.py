#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Daily ATR Breakout with Volume and Trend Filter
# Hypothesis: Price breaking above/below daily ATR-based bands with volume confirmation
# and trend filter (price vs 200 EMA) works in both bull and bear markets.
# In bull markets: buy on upward breaks above upper band, sell on downward breaks below lower band.
# In bear markets: sell on upward breaks above upper band, buy on downward breaks below lower band.
# Target: 15-30 trades/year (60-120 over 4 years).

name = "12h_daily_atr_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 14:
        return np.zeros(n)
    
    # Calculate daily ATR (14-period)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # True Range
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.roll(daily_close, 1))
    tr3 = np.abs(daily_low - np.roll(daily_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR using Wilder's smoothing (equivalent to RMA)
    atr = np.zeros_like(tr)
    atr[13] = np.mean(tr[0:14])  # First ATR value
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate bands: ±1.5 * ATR from close
    upper_band = daily_close + 1.5 * atr
    lower_band = daily_close - 1.5 * atr
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    upper_band = np.roll(upper_band, 1)
    lower_band = np.roll(lower_band, 1)
    upper_band[0] = upper_band[1] if len(upper_band) > 1 else 0
    lower_band[0] = lower_band[1] if len(lower_band) > 1 else 0
    
    # Align to 12h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_daily, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_daily, lower_band)
    
    # Trend filter: price vs 200 EMA
    close_series = pd.Series(close)
    ema_200 = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(ema_200[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: reversal below lower band or trend/volume failure
            if (low[i] <= lower_band_aligned[i] and close[i] < lower_band_aligned[i]) or \
               close[i] < ema_200[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit conditions: reversal above upper band or trend/volume failure
            if (high[i] >= upper_band_aligned[i] and close[i] > upper_band_aligned[i]) or \
               close[i] > ema_200[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: break above upper band with volume and trend
            if (high[i] > upper_band_aligned[i] and close[i] > upper_band_aligned[i]) and \
               close[i] > ema_200[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: break below lower band with volume and trend
            elif (low[i] < lower_band_aligned[i] and close[i] < lower_band_aligned[i]) and \
                 close[i] < ema_200[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals