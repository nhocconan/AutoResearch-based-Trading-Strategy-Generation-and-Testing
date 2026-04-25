#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout + 1d EMA34 Trend + Volume Spike + ATR Stoploss
Hypothesis: Camarilla pivot levels act as strong support/resistance. Breakouts above H3 or below L3
with 1d EMA34 trend alignment and volume confirmation capture high-probability momentum moves.
ATR-based stops limit drawdown. Works in bull markets via breakout continuation and in bear markets
via avoiding counter-trend breakouts. Discrete position sizing (0.25) minimizes fee churn.
Target: 20-50 trades/year on 4h timeframe.
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
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss
    if len(close) >= 14:
        tr1 = np.abs(np.diff(close, prepend=close[0]))
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr2[0] = np.abs(high[0] - close[0])
        tr3[0] = np.abs(low[0] - close[0])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.zeros(n)
        atr[:13] = np.nan
        for i in range(13, n):
            atr[i] = np.mean(tr[i-13:i+1])
    else:
        atr = np.full(n, np.nan)
    
    # Calculate 20-period volume MA for volume spike detection
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate Camarilla levels from previous 1d bar
    # We need previous day's OHLC - we'll calculate daily OHLC from 4h data
    # Group by day and get the OHLC for each day
    dates = pd.to_datetime(prices['open_time']).dt.date
    unique_dates = np.unique(dates)
    
    # Arrays to store daily OHLC aligned to each 4h bar
    prev_day_open = np.full(n, np.nan)
    prev_day_high = np.full(n, np.nan)
    prev_day_low = np.full(n, np.nan)
    prev_day_close = np.full(n, np.nan)
    
    for i in range(n):
        current_date = dates[i]
        # Find index of current date in unique_dates
        date_idx = np.where(unique_dates == current_date)[0]
        if len(date_idx) > 0 and date_idx[0] > 0:
            prev_date = unique_dates[date_idx[0] - 1]
            # Find all indices belonging to previous date
            prev_mask = (dates == prev_date)
            if np.any(prev_mask):
                prev_indices = np.where(prev_mask)[0]
                prev_day_open[i] = open_[prev_indices[0]]
                prev_day_high[i] = np.max(high[prev_indices])
                prev_day_low[i] = np.min(low[prev_indices])
                prev_day_close[i] = close[prev_indices[-1]]
    
    # Calculate Camarilla levels
    camarilla_H3 = np.full(n, np.nan)
    camarilla_L3 = np.full(n, np.nan)
    camarilla_H4 = np.full(n, np.nan)
    camarilla_L4 = np.full(n, np.nan)
    
    for i in range(n):
        if not (np.isnan(prev_day_high[i]) or np.isnan(prev_day_low[i]) or np.isnan(prev_day_close[i])):
            range_val = prev_day_high[i] - prev_day_low[i]
            camarilla_H3[i] = prev_day_close[i] + range_val * 1.1 / 4
            camarilla_L3[i] = prev_day_close[i] - range_val * 1.1 / 4
            camarilla_H4[i] = prev_day_close[i] + range_val * 1.1 / 2
            camarilla_L4[i] = prev_day_close[i] - range_val * 1.1 / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA34_1d, ATR, volume MA, and Camarilla to propagate
    start_idx = max(34, 14, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(camarilla_H3[i]) or 
            np.isnan(camarilla_L3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema34_1d = ema_34_1d_aligned[i]
        H3 = camarilla_H3[i]
        L3 = camarilla_L3[i]
        H4 = camarilla_H4[i]
        L4 = camarilla_L4[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long: price breaks above H3 AND uptrend (close > 1d EMA34) AND volume spike
            long_condition = (curr_close > H3) and (curr_close > ema34_1d) and volume_spike
            # Short: price breaks below L3 AND downtrend (close < 1d EMA34) AND volume spike
            short_condition = (curr_close < L3) and (curr_close < ema34_1d) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price breaks below L3 (reversal signal)
            if curr_close <= entry_price - 2.0 * atr_val or curr_close < L3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price breaks above H3 (reversal signal)
            if curr_close >= entry_price + 2.0 * atr_val or curr_close > H3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3_L3_Breakout_1dEMA34_Trend_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0