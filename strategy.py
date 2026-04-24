#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA34 trend filter and volume confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h EMA34 for trend direction (bullish when close > EMA34, bearish when close < EMA34).
- Session filter: 08-20 UTC to avoid low-volume Asian session noise.
- Entry: Price breaks above/below 1h Camarilla H3/L3 levels with volume > 1.5 * 20-period volume MA and 4h EMA34 alignment.
- Exit: ATR-based stoploss (1.5 * ATR(14)) or Camarilla level reversal (touch H4/L4).
- Signal size: 0.20 discrete to minimize fee churn.
Designed to work in both bull and bear markets by following higher timeframe trend while using lower timeframe breakouts for entry timing.
Session filter reduces whipsaw during low-liquidity periods.
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
    
    # Get 1h data for Camarilla calculation (using prior 1h bar's OHLC)
    # We need to calculate Camarilla levels based on previous bar's range
    if n < 2:
        return np.zeros(n)
    
    # Calculate 1h Camarilla levels from previous bar
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # first bar uses current close
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Camarilla levels calculation
    range_ = prev_high - prev_low
    camarilla_h4 = prev_close + 1.1 * range_ / 2
    camarilla_l4 = prev_close - 1.1 * range_ / 2
    camarilla_h3 = prev_close + 1.1 * range_ / 4
    camarilla_l3 = prev_close - 1.1 * range_ / 4
    
    # Get 4h data for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend
    close_4h = df_4h['close'].values
    ema_34 = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34)
    
    # Calculate 1h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1h volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready (need at least 2 for Camarilla, 34 for EMA, 20 for vol MA)
    start_idx = max(2, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]) or
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check session filter (08-20 UTC)
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold) and session
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            
            # Determine 4h EMA34 trend: bullish if close > EMA34, bearish if close < EMA34
            trend_bullish = close[i] > ema_34_aligned[i]
            trend_bearish = close[i] < ema_34_aligned[i]
            
            # Long: price breaks above Camarilla H3 AND 4h trend bullish AND volume confirmed AND in session
            if curr_high > camarilla_h3[i] and trend_bullish and vol_confirmed:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short: price breaks below Camarilla L3 AND 4h trend bearish AND volume confirmed AND in session
            elif curr_low < camarilla_l3[i] and trend_bearish and vol_confirmed:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on stoploss or price breaks below Camarilla L4 (reversal signal)
            stop_loss = entry_price - 1.5 * atr[i]
            if curr_low < stop_loss or curr_low < camarilla_l4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: exit on stoploss or price breaks above Camarilla H4 (reversal signal)
            stop_loss = entry_price + 1.5 * atr[i]
            if curr_high > stop_loss or curr_high > camarilla_h4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA34_Trend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0