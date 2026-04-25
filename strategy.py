#!/usr/bin/env python3
"""
12h Camarilla H3L3 Breakout with 1d EMA34 Trend Filter and Volume Spike Confirmation
Hypothesis: Camarilla pivot levels (H3/L3) act as strong intraday support/resistance.
Breakouts above H3 or below L3 with volume confirmation (>2.0x 24-bar vol MA) and
trend alignment (price > 1d EMA34 for longs, < 1d EMA34 for shorts) capture momentum.
In ranging markets, price tends to revert to the mean (pivot point). Designed for 12h
timeframe to avoid overtrading while profiting from both trending and mean-reverting
regimes across bull and bear markets. Target: 12-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need 34 for EMA + 1 for shift
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Typical price = (H + L + C) / 3
    # Range = H - L
    # H4 = C + 1.5 * (H - L)
    # H3 = C + 1.25 * (H - L)
    # L3 = C - 1.25 * (H - L)
    # L4 = C - 1.5 * (H - L)
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_1d = df_1d['high'] - df_1d['low']
    
    # Camarilla levels
    h3_1d = df_1d['close'] + 1.25 * range_1d
    l3_1d = df_1d['close'] - 1.25 * range_1d
    pp_1d = typical_price_1d  # Pivot point
    
    # Align to 12h timeframe (values from previous 1d bar)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d.values)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d.values)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d.values)
    
    # Calculate 24-period volume MA for volume spike confirmation (12h)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34, volume MA, and Camarilla levels
    start_idx = max(35, 24)  # 35 for EMA34 (34 + 1 for shift), 24 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(h3_1d_aligned[i]) or 
            np.isnan(l3_1d_aligned[i]) or 
            np.isnan(pp_1d_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_34_val = ema_34_1d_aligned[i]
        h3_val = h3_1d_aligned[i]
        l3_val = l3_1d_aligned[i]
        pp_val = pp_1d_aligned[i]
        vol_ma = vol_ma_24[i]
        
        # Volume confirmation: current volume > 2.0 * 24-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Breakout logic: enter on break of H3/L3 with volume and trend confirmation
            # Long: break above H3 + price > 1d EMA34 (uptrend) + volume confirmation
            long_breakout = (curr_high > h3_val) and (curr_close > ema_34_val) and volume_confirm
            # Short: break below L3 + price < 1d EMA34 (downtrend) + volume confirmation
            short_breakout = (curr_low < l3_val) and (curr_close < ema_34_val) and volume_confirm
            
            # Mean reversion logic: fade moves near extremes when ranging
            # Long: near L3 + price < 1d EMA34 (weak downtrend/ranging) + volume confirmation
            long_mean_revert = (curr_low <= l3_val * 1.002) and (curr_close < ema_34_val) and volume_confirm
            # Short: near H3 + price > 1d EMA34 (weak uptrend/ranging) + volume confirmation
            short_mean_revert = (curr_high >= h3_val * 0.998) and (curr_close > ema_34_val) and volume_confirm
            
            if long_breakout or long_mean_revert:
                signals[i] = 0.25
                position = 1
            elif short_breakout or short_mean_revert:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below pivot point OR breaks below L3 (failure)
            if curr_close < pp_val or curr_low < l3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above pivot point OR breaks above H3 (failure)
            if curr_close > pp_val or curr_high > h3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0