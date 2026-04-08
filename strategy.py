#!/usr/bin/env python3
# 12h_daily_camarilla_pivot_volume_regime_v1
# Hypothesis: 12h Camarilla pivot reversals with daily volume confirmation and 1d regime filter.
# Long: price touches Camarilla L3 support with volume > 1.8x 20-period average AND 1d close > 1d EMA50 (bullish regime)
# Short: price touches Camarilla H3 resistance with volume > 1.8x 20-period average AND 1d close < 1d EMA50 (bearish regime)
# Exit: price crosses Camarilla H4/L4 levels or ATR-based stoploss (2x ATR)
# Uses 12h primary timeframe with 1d HTF for EMA regime filter.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_camarilla_pivot_volume_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.mean(tr[i-14:i])
    
    # Calculate volume ratio (current vs 20-period average)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0)
    
    # Get 1d data for HTF regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA(50) on 1d for regime filter
    ema_50_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(df_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 / (50 + 1)) + (ema_50_1d[i-1] * (49 / (50 + 1)))
    
    # Align 1d EMA to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(50, n):
        vol_r = vol_ratio[i]
        price = close[i]
        
        if np.isnan(vol_r) or np.isnan(atr[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        regime_bullish = close[i] > ema_50_1d_aligned[i]
        regime_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Calculate Camarilla pivot levels using previous 12h bar's OHLC
        if i >= 1:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            pivot = (prev_high + prev_low + prev_close) / 3.0
            range_hl = prev_high - prev_low
            
            # Camarilla levels
            h4 = pivot + range_hl * 1.1 / 2
            h3 = pivot + range_hl * 1.1 / 4
            l3 = pivot - range_hl * 1.1 / 4
            l4 = pivot - range_hl * 1.1 / 2
        else:
            h4 = h3 = l3 = l4 = np.nan
        
        if position == 1:  # Long position
            # Exit: price crosses H4 OR stoploss hit (2x ATR below entry)
            if price >= h4 or price <= entry_price - 2.0 * atr_stop:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses L4 OR stoploss hit (2x ATR above entry)
            if price <= l4 or price >= entry_price + 2.0 * atr_stop:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: price touches L3 support with volume AND bullish regime
            if price <= l3 and vol_r > 1.8 and regime_bullish:
                position = 1
                entry_price = price
                atr_stop = atr[i]
                signals[i] = 0.25
            # Short entry: price touches H3 resistance with volume AND bearish regime
            elif price >= h3 and vol_r > 1.8 and regime_bearish:
                position = -1
                entry_price = price
                atr_stop = atr[i]
                signals[i] = -0.25
    
    return signals