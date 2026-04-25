#!/usr/bin/env python3
"""
4h Camarilla H3L3 Breakout with Volume Spike and 1d EMA34 Trend Filter - Enhanced
Hypothesis: Camarilla H3/L3 levels act as intraday support/resistance. Breakouts with volume confirmation indicate institutional participation. The 1d EMA34 trend filter ensures trades align with the daily trend, reducing false breakouts in choppy markets and improving performance in both bull and bear regimes. Added ATR-based stoploss to manage risk and discrete sizing (0.0, ±0.30) to balance performance and fee drag. Target: 20-40 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot calculation and EMA trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR for stoploss (using 4h data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla pivots from previous 1d OHLC
    # H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low), L3 = close - 1.0*(high-low), L4 = close - 1.5*(high-low)
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    rang = prev_high - prev_low
    H3 = prev_close + 1.0 * rang
    L3 = prev_close - 1.0 * rang
    
    # Align Camarilla levels to 4h (use previous day's levels for current day's trading)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for volume MA and EMA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        H3_level = H3_aligned[i]
        L3_level = L3_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 AND volume spike AND price > 1d EMA34 (uptrend)
            long_entry = (curr_close > H3_level) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below L3 AND volume spike AND price < 1d EMA34 (downtrend)
            short_entry = (curr_close < L3_level) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below L3 (reversal) OR price < 1d EMA34 (trend change) OR ATR stoploss hit
            if (curr_close < L3_level) or (curr_close < ema_trend) or (curr_low <= entry_price - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: price crosses above H3 (reversal) OR price > 1d EMA34 (trend change) OR ATR stoploss hit
            if (curr_close > H3_level) or (curr_close > ema_trend) or (curr_high >= entry_price + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_VolumeSpike_1dEMA34_Trend_ATR_SL_Enhanced"
timeframe = "4h"
leverage = 1.0