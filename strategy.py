#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R extreme with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA34 for trend direction (bullish when close > EMA34, bearish when close < EMA34).
- Entry: Long when Williams %R crosses above -80 (oversold) in 1d bull trend with volume > 1.8 * 4h volume MA(20); Short when Williams %R crosses below -20 (overbought) in 1d bear trend with volume > 1.8 * 4h volume MA(20).
- Exit: ATR-based trailing stop (2.0 * ATR(14)) or opposite Williams %R signal.
- Signal size: 0.25 discrete to balance capture and fee control.
- Designed for BTC/ETH: Williams %R captures mean reversion in extremes, EMA34 filter avoids counter-trend trades, volume confirmation ensures institutional participation, works in both bull and bear markets via trend-following logic.
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
    
    # Get 4h data for Williams %R calculation and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 4h volume MA(20) for confirmation
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Calculate Williams %R(14) on 4h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    wr = np.where((highest_high - lowest_low) == 0, -50, wr)
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0
    lowest_since_entry = 0
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_4h_aligned[i]) or 
            np.isnan(wr[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 1.8x threshold (balanced to reduce trades)
        vol_confirmed = curr_volume > 1.8 * vol_ma_4h_aligned[i]
        
        # Determine 1d EMA34 trend: bullish if close > EMA34, bearish if close < EMA34
        trend_bullish = close[i] > ema_34_aligned[i]
        trend_bearish = close[i] < ema_34_aligned[i]
        
        # Williams %R signals: long when crossing above -80 from below, short when crossing below -20 from above
        wr_long_signal = (wr[i-1] <= -80 and wr[i] > -80) if i > 0 else False
        wr_short_signal = (wr[i-1] >= -20 and wr[i] < -20) if i > 0 else False
        
        if position == 0:
            # Check for entry signals
            # Long: Williams %R crosses above -80 (oversold) in 1d bull trend with volume confirmation
            if wr_long_signal and trend_bullish and vol_confirmed:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            # Short: Williams %R crosses below -20 (overbought) in 1d bear trend with volume confirmation
            elif wr_short_signal and trend_bearish and vol_confirmed:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
        elif position == 1:
            # Long position: update highest and check exit conditions
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit: ATR trailing stop or opposite Williams %R signal (cross below -50)
            wr_exit_signal = (wr[i-1] >= -50 and wr[i] < -50) if i > 0 else False
            if curr_low <= highest_since_entry - 2.0 * atr[i] or wr_exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest and check exit conditions
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit: ATR trailing stop or opposite Williams %R signal (cross above -50)
            wr_exit_signal = (wr[i-1] <= -50 and wr[i] > -50) if i > 0 else False
            if curr_high >= lowest_since_entry + 2.0 * atr[i] or wr_exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0