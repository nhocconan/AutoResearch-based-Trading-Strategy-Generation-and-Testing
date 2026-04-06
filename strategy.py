#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Volume-Weighted Average Price (VWAP) mean reversion with daily EMA trend filter.
# Uses VWAP deviation from daily mean (20-period) as mean reversion signal.
# Daily EMA200 ensures trades align with long-term trend (buy dips in uptrend, sell rallies in downtrend).
# Volume spike filter (current volume > 2x 20-period average) confirms institutional interest.
# Designed for 6h timeframe to target 75-150 trades over 4 years.
# Works in bull/bear markets via daily EMA200 trend filter and mean reversion logic.

name = "6h_vwap_mean_reversion_daily_ema200_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily EMA200 for trend bias
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily closes
    ema_200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema_200_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema_200_1d[i] = (close_1d[i] * 2 / 201) + (ema_200_1d[i-1] * 199 / 201)
    
    # Align EMA200 to 6h timeframe (shifted by 1 daily bar for no look-ahead)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # VWAP calculation (typical price * volume) cumulative
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.divide(vwap_numerator, vwap_denominator, 
                     out=np.full_like(vwap_numerator, np.nan), 
                     where=vwap_denominator!=0)
    
    # VWAP deviation from 20-period mean
    vwap_ma = np.full(n, np.nan)
    vwap_std = np.full(n, np.nan)
    for i in range(19, n):
        vwap_slice = vwap[i-19:i+1]
        if not np.all(np.isnan(vwap_slice)):
            vwap_ma[i] = np.nanmean(vwap_slice)
            vwap_std[i] = np.nanstd(vwap_slice)
    
    # VWAP z-score (deviation in standard deviations)
    vwap_zscore = np.full(n, np.nan)
    for i in range(19, n):
        if not np.isnan(vwap_ma[i]) and vwap_std[i] > 0:
            vwap_zscore[i] = (vwap[i] - vwap_ma[i]) / vwap_std[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_200_aligned[i]) or np.isnan(vwap_zscore[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 2x 20-period average
        vol_ma = np.mean(volume[max(0, i-19):i+1]) if i >= 19 else np.nan
        if np.isnan(vol_ma):
            volume_filter = False
        else:
            volume_filter = volume[i] > vol_ma * 2.0
        
        # Trend bias: daily EMA200
        bullish_bias = close[i] > ema_200_aligned[i]
        bearish_bias = close[i] < ema_200_aligned[i]
        
        # Mean reversion signals
        # Long when VWAP significantly below mean (-2 sigma) in uptrend
        long_signal = vwap_zscore[i] < -2.0 and bullish_bias
        # Short when VWAP significantly above mean (+2 sigma) in downtrend
        short_signal = vwap_zscore[i] > 2.0 and bearish_bias
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: VWAP reverts to mean or stoploss (2x ATR approximation)
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.0 * atr_approx
            
            if (vwap_zscore[i] > -0.5 or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: VWAP reverts to mean or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.0 * atr_approx
            
            if (vwap_zscore[i] < 0.5 or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if volume_filter:
                if long_signal:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif short_signal:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals