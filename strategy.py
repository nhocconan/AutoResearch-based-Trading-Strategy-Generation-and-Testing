#!/usr/bin/env python3
"""
4h_VWAP_MeanReversion_TrendFilter_Volume
Hypothesis: Price reverts to VWAP (volume-weighted average price) from the prior day, with trend filter from daily EMA and volume confirmation, works in both bull and bear markets by capturing intraday mean reversion while respecting higher timeframe direction. Low frequency via 4h timeframe and strict entry criteria.
Target: 75-200 total trades over 4 years (19-50/year).
"""
name = "4h_VWAP_MeanReversion_TrendFilter_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for prior day VWAP and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate prior day VWAP
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d_prev = vwap_1d.shift(1).values  # Use prior day's VWAP
    
    # Align prior day VWAP to 4h timeframe (available after daily bar closes)
    vwap_1d_prev_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d_prev)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.5 * 24-period average (6h equivalent)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    # 4-period RSI for mean reversion confirmation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    avg_loss = loss.ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Need volume average and aligned data
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(vwap_1d_prev_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(rsi_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price below prior day VWAP + 1d uptrend + volume + RSI < 40 (oversold)
            if close[i] < vwap_1d_prev_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_filter[i] and rsi_values[i] < 40:
                signals[i] = 0.25
                position = 1
            # Short: price above prior day VWAP + 1d downtrend + volume + RSI > 60 (overbought)
            elif close[i] > vwap_1d_prev_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_filter[i] and rsi_values[i] > 60:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to prior day VWAP (mean reversion target)
            if position == 1:
                if close[i] >= vwap_1d_prev_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] <= vwap_1d_prev_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals