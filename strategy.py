#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Bollinger Band breakout with weekly trend filter and volume confirmation.
# Uses daily Bollinger Bands (20,2) for volatility bands and weekly EMA(34) for trend direction.
# Enters on daily close outside Bollinger Bands with volume confirmation and weekly trend alignment.
# Designed for low trade frequency (target 10-25/year) to minimize fee drag in both bull and bear markets.
# Works in bull markets by catching breakouts and in bear markets by fading overextended moves.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands (20,2) on daily data
    sma_20 = np.full(len(close_1d), np.nan)
    std_20 = np.full(len(close_1d), np.nan)
    for i in range(20, len(close_1d)):
        sma_20[i] = np.mean(close_1d[i-20:i])
        std_20[i] = np.std(close_1d[i-20:i])
    
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    
    # Align Bollinger Bands to daily timeframe (no alignment needed as we're using daily data)
    upper_band_aligned = upper_band
    lower_band_aligned = lower_band
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(34) on weekly close
    ema_34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema_34_1w[33] = np.mean(close_1w[:34])  # SMA for first value
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            ema_34_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_34_1w[i-1]
    
    # Align weekly EMA to daily timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate daily ATR for position sizing and stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.full(n, np.nan)
    if len(tr) >= 14:
        atr[13] = np.mean(tr[:14])  # SMA for first value
        alpha = 2 / (14 + 1)
        for i in range(14, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 20)  # need weekly EMA, Bollinger Bands, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above weekly EMA34 (uptrend) or below (downtrend)
        trend_up = close[i] > ema_34_1w_aligned[i]
        trend_down = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long entry: close above upper Bollinger Band with volume and uptrend
            if (close[i] > upper_band_aligned[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: close below lower Bollinger Band with volume and downtrend
            elif (close[i] < lower_band_aligned[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below middle Bollinger Band (SMA20)
            if close[i] < sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above middle Bollinger Band (SMA20)
            if close[i] > sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_BollingerBreakout_WeeklyEMA34_VolumeFilter"
timeframe = "1d"
leverage = 1.0