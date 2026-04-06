#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour VWAP Reversion with Daily Trend Filter.
# Uses VWAP deviation for mean reversion entries when price deviates >1.5 std from VWAP.
# Trend filter from daily EMA50 ensures trades align with higher timeframe direction.
# Volume filter (current volume > 1.2x 20-period average) ensures quality signals.
# Works in both bull/bear markets: mean reversion in range, trend-following in strong moves.
# Target: 75-200 trades over 4 years (19-50/year).

name = "6h_vwap_reversion_trend_v1"
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
    
    # Daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily close
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema50_1d[i] = (close_1d[i] * 2/51) + (ema50_1d[i-1] * 49/51)
    
    # Align daily EMA50 to 6h timeframe (shifted by 1 daily bar)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # VWAP calculation (typical price * volume)
    typical_price = (high + low + close) / 3.0
    vp = typical_price * volume
    
    # Cumulative VWAP with 20-period window
    cum_vp = np.full(n, np.nan)
    cum_vol = np.full(n, np.nan)
    vwap = np.full(n, np.nan)
    
    for i in range(n):
        start_idx = max(0, i - 19)
        cum_vp[i] = np.sum(vp[start_idx:i+1])
        cum_vol[i] = np.sum(volume[start_idx:i+1])
        if cum_vol[i] > 0:
            vwap[i] = cum_vp[i] / cum_vol[i]
    
    # VWAP deviation standard deviation (20-period)
    vwap_dev = np.full(n, np.nan)
    vwap_ma = np.full(n, np.nan)
    
    for i in range(19, n):
        if not np.isnan(vwap[i]):
            dev = close[i] - vwap[i]
            vwap_dev[i] = dev
            # Calculate rolling std of deviation
            start_idx = max(0, i - 19)
            dev_slice = vwap_dev[start_idx:i+1]
            valid_dev = dev_slice[~np.isnan(dev_slice)]
            if len(valid_dev) >= 2:
                vwap_ma[i] = np.mean(valid_dev)
                if len(valid_dev) >= 5:
                    vwap_std = np.std(valid_dev)
                else:
                    vwap_std = 0.001
            else:
                vwap_ma[i] = 0.0
                vwap_std = 0.001
    
    # Volume filter: current volume > 1.2x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(ema50_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(vwap_dev[i]) or np.isnan(vwap_ma[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.2
        
        # VWAP z-score (deviation / std)
        if vwap_std > 0:
            z_score = (close[i] - vwap[i]) / vwap_std
        else:
            z_score = 0.0
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price returns to VWAP or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.0 * atr_approx
            
            if (close[i] <= vwap[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to VWAP or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.0 * atr_approx
            
            if (close[i] >= vwap[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if volume_filter:
                # Mean reversion: fade extreme VWAP deviation
                # Short when price is significantly above VWAP and trend is down
                if (z_score > 1.5 and close[i] < ema50_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                # Long when price is significantly below VWAP and trend is up
                elif (z_score < -1.5 and close[i] > ema50_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
    
    return signals