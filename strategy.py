#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hTrend_ATRStop_v1
Hypothesis: Camarilla R3/S3 breakouts on 1h filtered by 4h EMA34 trend for directional bias.
Only take longs when price > 4h EMA34 (bullish bias) and shorts when price < 4h EMA34 (bearish bias).
Volume confirmation (>1.5x 20-period average) avoids false breakouts. ATR-based stoploss with 1.5x ATR.
Designed for 15-37 trades/year per symbol (~60-150 total over 4 years) to minimize fee drag.
Works in bull/bear via 4h trend alignment as regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for EMA34 trend filter)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # === 1h OHLC for Camarilla calculation (using previous day's data) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate previous day's OHLC for Camarilla levels
    # We need to group by day to get prior day's H/L/C
    df = prices.copy()
    df['date'] = df['open_time'].dt.date
    # Get prior day's high, low, close for each bar
    prev_high = df.groupby('date')['high'].shift(1).values
    prev_low = df.groupby('date')['low'].shift(1).values
    prev_close = df.groupby('date')['close'].shift(1).values
    
    # Camarilla levels: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4.0
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4.0
    
    # === 4h EMA34 for trend filter ===
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # === ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) 
            or np.isnan(ema_34_4h_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume filter: current volume > 1.5x 20-period average
            volume = prices['volume'].values
            vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
            vol_filter = volume[i] > 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
            
            # Long conditions: price > Camarilla R3, 4h uptrend, volume filter
            long_breakout = price > camarilla_r3[i]
            long_trend = price > ema_34_4h_aligned[i]
            
            # Short conditions: price < Camarilla S3, 4h downtrend, volume filter
            short_breakout = price < camarilla_s3[i]
            short_trend = price < ema_34_4h_aligned[i]
            
            # Entry logic - ONLY enter on volume filter + trend alignment
            if long_breakout and long_trend and vol_filter:
                signals[i] = 0.20
                position = 1
                entry_price = price
            elif short_breakout and short_trend and vol_filter:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below Camarilla S3 (support broken)
            elif price < camarilla_s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above Camarilla R3 (resistance broken)
            elif price > camarilla_r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_ATRStop_v1"
timeframe = "1h"
leverage = 1.0