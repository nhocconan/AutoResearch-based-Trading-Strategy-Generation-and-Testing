#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla pivot (H3/L3) breakout with 4h EMA(34) trend filter and volume confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h EMA(34) for trend filter (defines bull/bear regime).
- Entry: Long when price breaks above Camarilla H3 in bull regime with volume > 1.5 * 1h volume MA(20);
         Short when price breaks below Camarilla L3 in bear regime with volume > 1.5 * 1h volume MA(20).
- Exit: Price crosses below/above Camarilla H4/L4 levels (closer to pivot for faster reversion).
- Session filter: 08:00-20:00 UTC only to reduce noise trades.
- Signal size: 0.20 discrete to minimize fee churn while allowing meaningful position.
- Works in bull (buying H3 breakouts in uptrend) and bear (selling L3 breakdowns in downtrend).
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
    
    # Get 4h data for EMA calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h EMA(34)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1h volume MA(20) for confirmation
    vol_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Pre-compute session hours for efficiency
    hours = prices.index.hour if hasattr(prices.index, 'hour') else pd.DatetimeIndex(prices['open_time']).hour
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA needs 34, volume MA needs 20
    
    for i in range(start_idx, n):
        # Session filter: 08:00-20:00 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if data not ready
        if np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma_1h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Calculate Camarilla pivot levels using previous day's OHLC
        # Need to get daily OHLC - use 1d data from mtf_data
        if i >= 24:  # Need at least 24 hours of 1h data for previous day
            # Get daily OHLC from 1h data (simplified: use rolling window)
            # For proper Camarilla, we need actual daily OHLC
            # Approximate using 24-period lookback for 1h timeframe
            lookback_24 = max(0, i-23)
            prev_high = np.max(high[lookback_24:i])  # Previous period high (exclude current)
            prev_low = np.min(low[lookback_24:i])    # Previous period low
            prev_close = close[i-1]                  # Previous close
            
            # Calculate pivot point
            pivot = (prev_high + prev_low + prev_close) / 3.0
            range_hl = prev_high - prev_low
            
            # Camarilla levels
            h3 = pivot + (range_hl * 1.1 / 4.0)
            l3 = pivot - (range_hl * 1.1 / 4.0)
            h4 = pivot + (range_hl * 1.1 / 2.0)
            l4 = pivot - (range_hl * 1.1 / 2.0)
        else:
            # Not enough data for pivot calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: 1.5x threshold (balanced for 1h timeframe)
        vol_confirm = curr_volume > 1.5 * vol_ma_1h[i]
        
        # Trend filter: price relative to 4h EMA
        bull_regime = curr_close > ema_4h_aligned[i]
        bear_regime = curr_close < ema_4h_aligned[i]
        
        if position == 0:
            # Check for entry signals
            # Long: price breaks above Camarilla H3 in bull regime with volume confirmation
            if curr_high > h3 and bull_regime and vol_confirm:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla L3 in bear regime with volume confirmation
            elif curr_low < l3 and bear_regime and vol_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: exit when price crosses below Camarilla H4
            if curr_close < h4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: exit when price crosses above Camarilla L4
            if curr_close > l4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA34_Trend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0