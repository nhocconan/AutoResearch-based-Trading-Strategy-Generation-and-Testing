#!/usr/bin/env python3
"""
1d Camarilla H3/L3 Breakout + 1w EMA34 Trend + Volume Spike
Hypothesis: On daily timeframe, Camarilla H3/L3 levels represent strong intraday support/resistance derived from prior day's range. Breakouts above H3 or below L3 with weekly EMA34 trend alignment capture sustained momentum moves. Volume spike confirms institutional participation. Works in bull markets via buying H3 breakouts, bear markets via selling L3 breakdowns. Discrete position sizing (0.25) controls drawdown. Target: 15-25 trades/year on 1d.
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
    
    # Get 1w data for EMA34 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR(14) for stoploss
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Pre-compute 20-period volume MA for volume spike detection
    vol_ma_20 = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for ATR(14) and EMA34 to propagate
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_34 = ema_34_1w_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Calculate Camarilla pivot levels for today (using previous day's OHLC)
        # We need to get the previous day's high, low, close
        if i >= 1:  # Need at least one previous day
            # Get index of previous day's OHLC (24h ago for 1d timeframe)
            prev_day_idx = i - 1
            if prev_day_idx >= 0:
                # Get high, low, close of the previous day
                day_high = high[prev_day_idx]
                day_low = low[prev_day_idx]
                day_close = close[prev_day_idx]
                
                # Calculate Camarilla levels
                range_val = day_high - day_low
                if range_val > 0:
                    camarilla_h3 = day_close + (range_val * 1.1 / 4)
                    camarilla_l3 = day_close - (range_val * 1.1 / 4)
                else:
                    camarilla_h3 = curr_high
                    camarilla_l3 = curr_low
            else:
                camarilla_h3 = curr_high
                camarilla_l3 = curr_low
        else:
            camarilla_h3 = curr_high
            camarilla_l3 = curr_low
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long: break above Camarilla H3 AND uptrend AND volume spike
            long_condition = curr_close > camarilla_h3 and curr_close > ema_34 and volume_spike
            # Short: break below Camarilla L3 AND downtrend AND volume spike
            short_condition = curr_close < camarilla_l3 and curr_close < ema_34 and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price falls below EMA34
            if curr_close <= entry_price - 2.0 * atr_val or curr_close < ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price rises above EMA34
            if curr_close >= entry_price + 2.0 * atr_val or curr_close > ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0