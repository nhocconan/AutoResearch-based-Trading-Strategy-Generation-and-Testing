#!/usr/bin/env python3
"""
4h_Camarilla_R2_S2_SupportResistance_Trend_Filter_v1
Hypothesis: Trade 4h timeframe using Camarilla R2/S2 levels (stronger support/resistance) with 1d EMA50 trend filter and volume confirmation.
In uptrend: buy R2 breakout. In downtrend: sell S2 breakdown. In range: fade S2/R2 with confirmation.
Uses fewer trades than R1/S1 strategy to reduce fee drag while maintaining edge.
Works in bull markets via breakouts and bear via breakdowns/mean reversion at stronger levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate CAMARILLA levels from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for CAMARILLA calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # CAMARILLA R2 and S2 levels (stronger support/resistance)
    camarilla_r2 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s2 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align CAMARILLA levels to 4h timeframe
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    
    # 1d EMA50 for trend filter (smoother than EMA34)
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: current volume > 1.8 * 20-period average (on 4h data, ~3.3 days)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for volume average and EMA
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data not ready
        if (np.isnan(camarilla_r2_aligned[i]) or np.isnan(camarilla_s2_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        camarilla_r2_val = camarilla_r2_aligned[i]
        camarilla_s2_val = camarilla_s2_aligned[i]
        ema_50_val = ema_50_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Require minimum 16 bars since last exit to avoid churn (~2.6 days on 4h)
            if bars_since_exit >= 16:
                # Long: price breaks above R2 with volume confirmation AND above 1d EMA50 (uptrend)
                if close[i] > camarilla_r2_val and vol_conf and close[i] > ema_50_val:
                    signals[i] = size
                    position = 1
                    bars_since_exit = 0
                # Short: price breaks below S2 with volume confirmation AND below 1d EMA50 (downtrend)
                elif close[i] < camarilla_s2_val and vol_conf and close[i] < ema_50_val:
                    signals[i] = -size
                    position = -1
                    bars_since_exit = 0
                # Mean reversion in ranging markets: buy near S2, sell near R2
                elif abs(close[i] - camarilla_s2_val) < 0.001 * close[i] and vol_conf and close[i] < ema_50_val:
                    # Near S2 in downtrend - potential bounce
                    signals[i] = size * 0.5  # Half position for mean reversion
                    position = 1
                    bars_since_exit = 0
                elif abs(close[i] - camarilla_r2_val) < 0.001 * close[i] and vol_conf and close[i] > ema_50_val:
                    # Near R2 in uptrend - potential pullback
                    signals[i] = -size * 0.5  # Half position for mean reversion
                    position = -1
                    bars_since_exit = 0
        elif position == 1:
            # Exit long: price breaks below S2 (strong support) or reaches R2 (profit target)
            if close[i] < camarilla_s2_val:
                signals[i] = 0.0
                position = 0
            elif close[i] > camarilla_r2_val:
                signals[i] = 0.0  # Take profit at R2
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above R2 (strong resistance) or reaches S2 (profit target)
            if close[i] > camarilla_r2_val:
                signals[i] = 0.0
                position = 0
            elif close[i] < camarilla_s2_val:
                signals[i] = 0.0  # Take profit at S2
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R2_S2_SupportResistance_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0