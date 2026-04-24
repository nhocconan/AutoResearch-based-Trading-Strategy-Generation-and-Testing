#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h EMA50 for trend direction (bullish when price > EMA50, bearish when price < EMA50).
- Entry: Price breaks above/below 1h Camarilla H3/L3 levels with volume > 2.0 * 20-period volume MA and 4h EMA50 alignment.
- Exit: ATR-based stoploss (2.5 * ATR(14)) or Camarilla level reversal (opposite level touch).
- Signal size: 0.20 discrete to minimize fee churn and control drawdown.
- Session filter: Only trade between 08:00-20:00 UTC to avoid low-volume, noisy sessions.
- Added 4h volume confirmation: require 4h volume > 1.5 * 20-period 4h volume MA to ensure HTF participation.
Designed to capture strong 1h momentum moves at key intraday pivot levels with volume confirmation and trend filtering.
Camarilla levels provide precise intraday support/resistance that works in both bull and bear markets.
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
    
    # Get 1d data for Camarilla levels (H3, L3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Get 4h data for EMA50 and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Calculate 4h volume MA(20) for HTF volume confirmation
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Calculate 1d Camarilla levels (H3, L3)
    # Camarilla: H3 = close + 1.1*(high-low)*1.1/4, L3 = close - 1.1*(high-low)*1.1/4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 1h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1h volume MA(20) for LTF volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_50_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(vol_ma_4h_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold for LTF, 1.5x for HTF)
            vol_confirmed_ltf = curr_volume > 2.0 * vol_ma[i]
            vol_confirmed_htf = volume_4h[i // 16] > 1.5 * vol_ma_4h_aligned[i] if i >= 16 else False
            
            # Determine 4h EMA50 trend: bullish if price > EMA50, bearish if price < EMA50
            htf_close_aligned = align_htf_to_ltf(prices, df_4h, close_4h)
            htf_close = htf_close_aligned[i]
            
            trend_bullish = htf_close > ema_50_aligned[i]
            trend_bearish = htf_close < ema_50_aligned[i]
            
            # Long: price breaks above Camarilla H3 AND 4h trend bullish AND volume confirmed
            if curr_high > camarilla_h3_aligned[i] and trend_bullish and vol_confirmed_ltf and vol_confirmed_htf:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short: price breaks below Camarilla L3 AND 4h trend bearish AND volume confirmed
            elif curr_low < camarilla_l3_aligned[i] and trend_bearish and vol_confirmed_ltf and vol_confirmed_htf:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on stoploss or price breaks below Camarilla L3 (reversal signal)
            stop_loss = entry_price - 2.5 * atr[i]
            if curr_low < stop_loss or curr_low < camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: exit on stoploss or price breaks above Camarilla H3 (reversal signal)
            stop_loss = entry_price + 2.5 * atr[i]
            if curr_high > stop_loss or curr_high > camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA50_Trend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0