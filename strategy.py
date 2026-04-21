#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_Squeeze_Breakout_WeeklyTrend
Hypothesis: Daily Camarilla R3/S3 breakout with weekly EMA50 trend filter and Bollinger Band squeeze confirmation.
Works in bull/bear by requiring alignment with weekly trend and low volatility breakout (squeeze) for high-probability moves.
Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (weekly for EMA trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Weekly EMA50 for trend regime ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Daily ATR (10-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=10, min_periods=10).mean().values
    
    # === Bollinger Band (20,2) squeeze: BBW < 20th percentile of last 50 days ===
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(window=20, min_periods=20).mean()
    bb_std = close_s.rolling(window=20, min_periods=20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid
    bb_width_percentile = bb_width.rolling(window=50, min_periods=20).rank(pct=True)
    squeeze = bb_width_percentile < 0.2  # BBW in lowest 20%
    
    # === Daily Camarilla pivot levels (R3, S3) ===
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # first bar invalid
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r3 = pivot + (prev_high - prev_low) * 1.1 / 4.0
    s3 = pivot - (prev_high - prev_low) * 1.1 / 4.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(squeeze[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        squeeze_now = squeeze.iloc[i] if hasattr(squeeze, 'iloc') else squeeze[i]
        r3_val = r3[i]
        s3_val = s3[i]
        
        if position == 0:
            # Long: price breaks above R3, above weekly EMA50, Bollinger squeeze
            long_condition = (price > r3_val) and (price > ema_50_1w_val) and squeeze_now
            # Short: price breaks below S3, below weekly EMA50, Bollinger squeeze
            short_condition = (price < s3_val) and (price < ema_50_1w_val) and squeeze_now
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 3 days to reduce churn
            if bars_since_entry < 3:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (1.5x ATR)
            if position == 1:
                if price < entry_price - 1.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price below weekly EMA50)
                elif price < ema_50_1w_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 1.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price above weekly EMA50)
                elif price > ema_50_1w_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_Pivot_Squeeze_Breakout_WeeklyTrend"
timeframe = "1d"
leverage = 1.0