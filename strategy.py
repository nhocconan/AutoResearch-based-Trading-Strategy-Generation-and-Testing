#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when price breaks above H3 AND price > 4h EMA50 AND volume > 2.0 * 1h volume MA(20);
         Short when price breaks below L3 AND price < 4h EMA50 AND volume > 2.0 * 1h volume MA(20).
- Exit: Close below/above L3/H3 for profit-taking, with ATR-based stoploss (2.0 * ATR(14)).
- Signal size: 0.20 discrete to control fee drag.
- Uses Camarilla pivot points for intraday support/resistance, volume confirmation for participation,
  4h EMA50 trend filter to avoid counter-trend trades, and ATR for risk management.
- Designed to work in both bull and bear markets via trend filter and breakout logic.
- Session filter: 08-20 UTC to reduce noise trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1h Camarilla pivot points (based on previous day's OHLC)
    # We need to group by day to get previous day's OHLC
    # Since we don't have easy daily grouping, we'll use rolling window approximation
    # For Camarilla, we need: H3 = C + 1.1*(H-L)/2, L3 = C - 1.1*(H-L)/2
    # where C, H, L are from previous day
    # We'll approximate using 24-period lookback (24h = 1 day)
    lookback = 24  # 24 * 1h = 1 day
    if n < lookback + 1:
        return np.zeros(n)
    
    # Calculate previous day's OHLC using rolling window
    prev_day_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    prev_day_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    prev_day_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).last().shift(1).values
    
    # Calculate Camarilla levels
    H3 = prev_day_close + 1.1 * (prev_day_high - prev_day_low) / 2
    L3 = prev_day_close - 1.1 * (prev_day_high - prev_day_low) / 2
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h volume MA(20) for volume confirmation
    vol_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) for 1h timeframe for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high[0] - low[0]], tr])  # first TR is high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(lookback + 1, 50, 20, 14)  # lookback+1 for Camarilla, 50 for EMA50, 20 for vol MA, 14 for ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_1h[i]) or 
            np.isnan(atr14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr14[i]
        
        # Volume confirmation: 2.0x threshold
        vol_confirm = curr_volume > 2.0 * vol_ma_1h[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: price breaks above H3 AND price > 4h EMA50 (uptrend)
                if curr_high > H3[i] and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Short: price breaks below L3 AND price < 4h EMA50 (downtrend)
                elif curr_low < L3[i] and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        elif position == 1:
            # Long position: check exit conditions
            # Stoploss: 2.0 * ATR below entry
            stoploss = entry_price - 2.0 * curr_atr
            # Profit take: close below L3 (mean reversion to pivot)
            if curr_close < stoploss or curr_close < L3[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: check exit conditions
            # Stoploss: 2.0 * ATR above entry
            stoploss = entry_price + 2.0 * curr_atr
            # Profit take: close above H3 (mean reversion to pivot)
            if curr_close > stoploss or curr_close > H3[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0