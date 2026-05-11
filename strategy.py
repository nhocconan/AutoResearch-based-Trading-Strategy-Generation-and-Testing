#!/usr/bin/env python3
name = "1h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Session"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter (updated only after 1d bar closes)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1h OHLCV
    close_1h = prices['close'].values
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    volume_1h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA34 ---
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- 1h Camarilla Levels (based on previous 1d) ---
    # Calculate from previous 1d bar (shifted by 1 to avoid lookahead)
    prev_close = np.roll(close_1h, 1)
    prev_high = np.roll(high_1h, 1)
    prev_low = np.roll(low_1h, 1)
    prev_close[0] = close_1h[0]
    prev_high[0] = high_1h[0]
    prev_low[0] = low_1h[0]
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # --- Volume Filter: above 1.5x median of last 24 periods ---
    vol_median = pd.Series(volume_1h).rolling(window=24, min_periods=12).median().values
    vol_threshold = vol_median * 1.5
    
    # --- ATR for stoploss (14-period) ---
    tr1 = np.abs(high_1h - low_1h)
    tr2 = np.abs(high_1h - np.roll(close_1h, 1))
    tr3 = np.abs(low_1h - np.roll(close_1h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # --- Session filter: 08-20 UTC ---
    hours = prices.index.hour  # already datetime64[ms], .hour works
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 34  # for EMA34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_threshold[i]) or np.isnan(atr[i])):
            if position != 0:
                # Check stoploss
                if position == 1 and close_1h[i] <= entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_1h[i] >= entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20 if position == 1 else -0.20
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Determine 1d trend
        trend_up = close_1h[i] > ema34_1d_aligned[i]
        trend_down = close_1h[i] < ema34_1d_aligned[i]
        
        # Volume filter: above 1.5x median
        vol_ok = volume_1h[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries only in direction of 1d trend with volume and session
            if in_session and close_1h[i] > camarilla_r3[i] and trend_up and vol_ok:
                # Long: price breaks above R3 + 1d uptrend + volume + session
                signals[i] = 0.20
                position = 1
                entry_price = close_1h[i]
            elif in_session and close_1h[i] < camarilla_s3[i] and trend_down and vol_ok:
                # Short: price breaks below S3 + 1d downtrend + volume + session
                signals[i] = -0.20
                position = -1
                entry_price = close_1h[i]
        else:
            # Update stoploss and check exits
            if position == 1:
                # Stoploss
                if close_1h[i] <= entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses below S3
                elif close_1h[i] <= camarilla_s3[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Stoploss
                if close_1h[i] >= entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses above R3
                elif close_1h[i] >= camarilla_r3[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals