#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA34 trend filter and volume confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h EMA34 for trend direction (bullish when close > EMA34, bearish when close < EMA34).
- Entry: Price breaks above/below 1h Camarilla H3/L3 levels with volume > 2.0 * 20-period volume MA and 4h EMA34 alignment.
- Session filter: Only trade between 08:00-20:00 UTC to avoid low-liquidity periods.
- Exit: ATR-based stoploss (1.5 * ATR(14)) or Camarilla level reversal (touch opposite level).
- Signal size: 0.20 discrete to minimize fee churn while maintaining exposure.
Designed to work in both bull and bear markets by following 4h trend while using 1h Camarilla breakouts for precise entry timing.
Volume spike filter (2.0x) reduces false breakouts in choppy markets. Session filter avoids Asian session noise.
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
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1h data for Camarilla levels (using typical pivot calculation from previous bar)
    # For simplicity, we'll use rolling high/low/close as proxy for pivot calculation
    # In practice, Camarilla uses previous day's range, but we approximate with 1h structure
    high_1h = high
    low_1h = low
    close_1h = close
    
    # Calculate approximate Camarilla levels for 1h (H3, L3, H4, L4)
    # Using rolling window of 26 periods (1 day + 2h) to simulate daily pivot
    lookback = 26
    rolling_high = pd.Series(high_1h).rolling(window=lookback, min_periods=lookback).max().values
    rolling_low = pd.Series(low_1h).rolling(window=lookback, min_periods=lookback).min().values
    rolling_close = pd.Series(close_1h).rolling(window=lookback, min_periods=lookback).mean().values
    
    # Camarilla calculations based on previous period's range
    # H3 = close + 1.1*(high-low)/2
    # L3 = close - 1.1*(high-low)/2
    # H4 = close + 1.1*(high-low)
    # L4 = close - 1.1*(high-low)
    range_hl = rolling_high - rolling_low
    h3 = rolling_close + 1.1 * range_hl / 2
    l3 = rolling_close - 1.1 * range_hl / 2
    h4 = rolling_close + 1.1 * range_hl
    l4 = rolling_close - 1.1 * range_hl
    
    # Get 4h data for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34 = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34)
    
    # Calculate 1h ATR(14) for stoploss
    tr1 = high_1h - low_1h
    tr2 = np.abs(high_1h - np.roll(close_1h, 1))
    tr3 = np.abs(low_1h - np.roll(close_1h, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_1h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1h = pd.Series(tr_1h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1h volume MA(20) for confirmation
    vol_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if not in trading session or data not ready
        if not in_session[i] or \
           np.isnan(h3[i]) or np.isnan(l3[i]) or np.isnan(h4[i]) or np.isnan(l4[i]) or \
           np.isnan(ema_34_aligned[i]) or np.isnan(atr_1h[i]) or np.isnan(vol_ma_1h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold) and session filter
            vol_confirmed = curr_volume > 2.0 * vol_ma_1h[i]
            
            # Determine 4h EMA34 trend: bullish if close > EMA34, bearish if close < EMA34
            trend_bullish = close[i] > ema_34_aligned[i]
            trend_bearish = close[i] < ema_34_aligned[i]
            
            # Long: price breaks above Camarilla H3 level AND 4h trend bullish AND volume confirmed
            if curr_high > h3[i] and trend_bullish and vol_confirmed:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short: price breaks below Camarilla L3 level AND 4h trend bearish AND volume confirmed
            elif curr_low < l3[i] and trend_bearish and vol_confirmed:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on stoploss or price breaks below Camarilla L3 level (reversal signal)
            stop_loss = entry_price - 1.5 * atr_1h[i]
            if curr_low < stop_loss or curr_low < l3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: exit on stoploss or price breaks above Camarilla H3 level (reversal signal)
            stop_loss = entry_price + 1.5 * atr_1h[i]
            if curr_high > stop_loss or curr_high > h3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA34_Trend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0