#!/usr/bin/env python3
"""
1h_hull_ma_rsi_filter_4h1d_trend
Hypothesis: Use Hull Moving Average (HMA) on 1h for entry signals, filtered by 4h RSI trend and 1d volume regime. Works in bull/bear markets by using RSI > 50 for long bias and RSI < 50 for short bias on 4h, while 1d volume filter ensures trades occur only during active market periods. Hull MA reduces lag and whipsaws compared to traditional MA.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_hull_ma_rsi_filter_4h1d_trend"
timeframe = "1h"
leverage = 1.0

def _wma(values, window):
    """Weighted Moving Average"""
    weights = np.arange(1, window + 1)
    return np.convolve(values, weights / weights.sum(), mode='same')

def _hull_ma(values, period):
    """Hull Moving Average"""
    half_period = max(2, period // 2)
    sqrt_period = max(2, int(np.sqrt(period)))
    
    wma_half = _wma(values, half_period)
    wma_full = _wma(values, period)
    
    raw_hma = 2 * wma_half - wma_full
    return _wma(raw_hma, sqrt_period)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for RSI trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate RSI on 4h close
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    
    # 1d data for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate volume moving average on 1d
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 1h timeframe
    rsi_4h_1h = align_htf_to_ltf(prices, df_4h, rsi_4h)
    vol_ma_1d_1h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Hull Moving Average on 1h close (period 16 for responsiveness)
    hull_ma = _hull_ma(close, 16)
    
    # Volume confirmation on 1h (current vs 20-period average)
    vol_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(hull_ma[i]) or np.isnan(rsi_4h_1h[i]) or 
            np.isnan(vol_ma_1d_1h[i]) or np.isnan(vol_ma_1h[i]) or
            vol_ma_1d_1h[i] <= 0 or vol_ma_1h[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume regime: trade only when 1d volume is above average (active market)
        vol_regime = volume_1d[-1] > vol_ma_1d_1h[i] if len(volume_1d) > 0 else False
        
        # Volume confirmation on 1h: current volume > 1.2x average
        vol_confirm = volume[i] > 1.2 * vol_ma_1h[i]
        
        # Hull MA direction
        price_above_hull = close[i] > hull_ma[i]
        price_below_hull = close[i] < hull_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price crosses below Hull MA
            if price_below_hull:
                exit_long = True
            # Exit if 4h RSI turns weak (< 40)
            elif rsi_4h_1h[i] < 40:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price crosses above Hull MA
            if price_above_hull:
                exit_short = True
            # Exit if 4h RSI turns strong (> 60)
            elif rsi_4h_1h[i] > 60:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Need volume regime and confirmation
            if not (vol_regime and vol_confirm):
                signals[i] = 0.0
                continue
            
            # Long entry conditions
            long_entry = False
            # Price crosses above Hull MA with bullish 4h RSI (> 50)
            if price_above_hull and close[i-1] <= hull_ma[i-1]:
                if rsi_4h_1h[i] > 50:
                    long_entry = True
            
            # Short entry conditions
            short_entry = False
            # Price crosses below Hull MA with bearish 4h RSI (< 50)
            if price_below_hull and close[i-1] >= hull_ma[i-1]:
                if rsi_4h_1h[i] < 50:
                    short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.20
            elif short_entry:
                position = -1
                signals[i] = -0.20
    
    return signals