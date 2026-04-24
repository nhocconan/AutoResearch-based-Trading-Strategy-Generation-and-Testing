#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA34 for trend direction (bullish when close > EMA34, bearish when close < EMA34).
- Entry: Long when Williams %R(14) crosses above -20 from below in 1d bull trend with volume > 1.5 * 6h volume MA(20); 
         Short when Williams %R(14) crosses below -80 from above in 1d bear trend with volume > 1.5 * 6h volume MA(20).
- Exit: ATR-based trailing stop (2.0 * ATR(14)) or opposite Williams %R signal.
- Signal size: 0.25 discrete to balance capture and fee control.
- Designed for BTC/ETH: Williams %R identifies overbought/oversold extremes, EMA34 filter avoids counter-trend trades, 
  volume confirmation ensures institutional participation, works in both bull and bear markets via mean-reversion logic at extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Williams %R calculation and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 6h volume MA(20) for confirmation
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Williams %R(14) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0
    lowest_since_entry = 0
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_6h_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 1.5x threshold
        vol_confirmed = curr_volume > 1.5 * vol_ma_6h_aligned[i]
        
        # Determine 1d EMA34 trend: bullish if close > EMA34, bearish if close < EMA34
        trend_bullish = close[i] > ema_34_aligned[i]
        trend_bearish = close[i] < ema_34_aligned[i]
        
        # Williams %R signals
        williams_r_curr = williams_r[i]
        williams_r_prev = williams_r[i-1] if i > 0 else williams_r[i]
        
        # Long signal: Williams %R crosses above -20 from below (oversold to normal)
        long_signal = (williams_r_prev <= -20 and williams_r_curr > -20 and 
                      trend_bullish and vol_confirmed)
        # Short signal: Williams %R crosses below -80 from above (overbought to normal)
        short_signal = (williams_r_prev >= -80 and williams_r_curr < -80 and 
                       trend_bearish and vol_confirmed)
        
        if position == 0:
            # Check for entry signals
            if long_signal:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            elif short_signal:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
        elif position == 1:
            # Long position: update highest and check exit conditions
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit: ATR trailing stop or opposite Williams %R signal
            if (curr_low <= highest_since_entry - 2.0 * atr[i] or 
                (williams_r_prev >= -80 and williams_r_curr < -80)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest and check exit conditions
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit: ATR trailing stop or opposite Williams %R signal
            if (curr_high >= lowest_since_entry + 2.0 * atr[i] or 
                (williams_r_prev <= -20 and williams_r_curr > -20)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0