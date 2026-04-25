#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Camarilla H3/L3 levels represent strong intraday support/resistance. Breakouts above H3 or below L3 with 1d EMA34 trend alignment capture momentum moves. Volume spike confirms institutional participation. Chop filter avoids whipsaws in ranging markets. Works in bull markets via buying H3 breakouts, bear markets via selling L3 breakdowns. Discrete position sizing (0.25) controls drawdown. Target: 25-40 trades/year on 4h.
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
    
    # Get 1d data for EMA34 trend filter and chop calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d chopiness index for regime filter
    if len(df_1d) >= 14:
        tr1 = pd.Series(df_1d['high']).diff().abs()
        tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift()).abs()
        tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift()).abs()
        tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
        high_low_1d = (pd.Series(df_1d['high']) - pd.Series(df_1d['low'])).abs().rolling(window=14, min_periods=14).sum().values
        chop_1d = 100 * np.log10(atr_1d / high_low_1d) / np.log10(14)
        chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    else:
        chop_1d_aligned = np.full(n, 50.0)
    
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
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_34 = ema_34_1d_aligned[i]
        atr_val = atr[i]
        chop_val = chop_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Calculate Camarilla pivot levels for today (using previous day's OHLC)
        # We need to get the previous day's high, low, close
        if i >= 96:  # 96 * 4h = 16 days, enough to get previous day
            # Get index of 4h bars that belong to previous day
            # Simplified: use 24 bars back (6 * 4h = 24h) for previous day's OHLC
            prev_day_idx = i - 24
            if prev_day_idx >= 24:  # ensure we have enough data for previous day
                # Get high, low, close of the previous day (24h period)
                day_high = np.max(high[prev_day_idx-24:prev_day_idx])
                day_low = np.min(low[prev_day_idx-24:prev_day_idx])
                day_close = close[prev_day_idx-1]  # close of previous day
                
                # Calculate Camarilla levels
                range_val = day_high - day_low
                if range_val > 0:
                    camarilla_h3 = day_close + (range_val * 1.1 / 2)
                    camarilla_l3 = day_close - (range_val * 1.1 / 2)
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
        
        # Chop filter: avoid ranging markets (chop > 61.8) and extreme trending (chop < 38.2)
        # We want moderate chop: 38.2 <= chop <= 61.8 for breakout trading
        chop_filter = (chop_val >= 38.2) and (chop_val <= 61.8)
        
        if position == 0:
            # Long: break above Camarilla H3 AND uptrend AND volume spike AND chop filter
            long_condition = curr_close > camarilla_h3 and curr_close > ema_34 and volume_spike and chop_filter
            # Short: break below Camarilla L3 AND downtrend AND volume spike AND chop filter
            short_condition = curr_close < camarilla_l3 and curr_close < ema_34 and volume_spike and chop_filter
            
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

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0