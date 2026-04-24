#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Camarilla levels: H3/L3 from prior 1h candle (using prior close to avoid look-ahead).
- Entry: Long when price breaks above prior H3 AND 4h EMA34 bullish AND volume > 2.0 * volume MA(20).
         Short when price breaks below prior L3 AND 4h EMA34 bearish AND volume > 2.0 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below 4h EMA34,
        exit short when price crosses above 4h EMA34.
- Signal size: 0.20 discrete to minimize fee churn and manage drawdown.
- Session filter: Only trade between 08:00-20:00 UTC to avoid low-liquidity hours.
This strategy captures intraday breakouts aligned with the 4h trend, designed to work in both bull and bear markets by following the intermediate trend.
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
    
    # Get 4h data for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 40:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend filter
    df_4h_close = df_4h['close'].values
    ema_4h = pd.Series(df_4h_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate prior 1h Camarilla H3/L3 levels (using prior candle to avoid look-ahead)
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    tp_series = pd.Series(typical_price)
    # Camarilla H3 = close + 1.1 * (high - low) / 6
    # Camarilla L3 = close - 1.1 * (high - low) / 6
    camarilla_high = tp_series + 1.1 * (pd.Series(high) - pd.Series(low)) / 6.0
    camarilla_low = tp_series - 1.1 * (pd.Series(high) - pd.Series(low)) / 6.0
    camarilla_high_shift = camarilla_high.shift(1).values
    camarilla_low_shift = camarilla_low.shift(1).values
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 1h
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 40)  # Need enough bars for EMA34 and Camarilla
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(camarilla_high_shift[i]) or 
            np.isnan(camarilla_low_shift[i]) or np.isnan(vol_ma[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold)
            vol_confirmed = curr_volume > 2.0 * vol_ma[i]
            
            # Long: Price breaks above prior H3 AND 4h EMA34 bullish AND volume confirmed
            if curr_close > camarilla_high_shift[i] and curr_close > ema_4h_aligned[i] and vol_confirmed:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below prior L3 AND 4h EMA34 bearish AND volume confirmed
            elif curr_close < camarilla_low_shift[i] and curr_close < ema_4h_aligned[i] and vol_confirmed:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long when price crosses below 4h EMA34 (trend change)
            if curr_close < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short when price crosses above 4h EMA34 (trend change)
            if curr_close > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA34_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0