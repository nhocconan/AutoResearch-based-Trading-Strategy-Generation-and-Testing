#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume_v3
Hypothesis: Trade 4h timeframe with daily CAMARILLA R1/S1 breakouts filtered by 1d EMA34 trend and volume spikes.
Adds ATR-based stoploss and minimum hold period to reduce trade frequency and improve risk-adjusted returns.
Target 15-25 trades/year to minimize fee drift. Works in bull markets via breakouts and bear via mean reversion at S1/R1 levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR for stoploss (using 4h data)
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[tr1[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros(n)
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # Calculate CAMARILLA levels from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for CAMARILLA calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # CAMARILLA R1 and S1 levels (core reversal levels)
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 6
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 6
    
    # Align CAMARILLA levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: current volume > 2.0 * 24-period average (on 4h data, ~4 days)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0
    size = 0.25   # Position size: 25% of capital
    min_hold_bars = 12  # Minimum 3 days before allowing reversal
    
    # Warmup: need enough data for volume average and EMA
    start_idx = max(24, 34)
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_confirm[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        ema_34_val = ema_34_aligned[i]
        vol_conf = volume_confirm[i]
        atr_val = atr[i]
        
        if position == 0:
            # Require minimum 12 bars since last exit to avoid churn (~2 days on 4h)
            if bars_since_exit >= min_hold_bars:
                # Long: price breaks above R1 with volume confirmation AND above 1d EMA34 (uptrend)
                if close[i] > camarilla_r1_val and vol_conf and close[i] > ema_34_val:
                    signals[i] = size
                    position = 1
                    bars_since_exit = 0
                # Short: price breaks below S1 with volume confirmation AND below 1d EMA34 (downtrend)
                elif close[i] < camarilla_s1_val and vol_conf and close[i] < ema_34_val:
                    signals[i] = -size
                    position = -1
                    bars_since_exit = 0
        elif position == 1:
            # Exit long: price breaks below S1 (opposite level) OR ATR-based stoploss
            if close[i] < camarilla_s1_val or close[i] < (high[i-1] - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above R1 (opposite level) OR ATR-based stoploss
            if close[i] > camarilla_r1_val or close[i] > (low[i-1] + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume_v3"
timeframe = "4h"
leverage = 1.0