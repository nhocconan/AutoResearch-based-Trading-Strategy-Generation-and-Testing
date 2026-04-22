#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot level breakout with 1d trend filter and volume confirmation
# Camarilla pivot levels (R1-S1) act as key support/resistance where breakouts often occur.
# Direction determined by 1d EMA34 trend (bullish if close > EMA34, bearish if close < EMA34).
# Entry confirmed by 4h volume spike (> 1.5x 20-bar average) to avoid false breakouts.
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for 4h timeframe targeting 20-40 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Load 4h data for Camarilla pivot and volume (ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Camarilla pivot levels from previous 4h bar
    # R1 = close + (high - low) * 1.1 / 12
    # S1 = close - (high - low) * 1.1 / 12
    camarilla_factor = (high_4h - low_4h) * 1.1 / 12
    r1_4h = close_4h + camarilla_factor
    s1_4h = close_4h - camarilla_factor
    
    # Align Camarilla levels to 4h chart (use previous bar's levels)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # 4h volume 20-period average for spike detection
    vol_avg_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or
            np.isnan(vol_avg_20_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + 1d uptrend + 4h volume spike
            if (close[i] > r1_4h_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20_4h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + 1d downtrend + 4h volume spike
            elif (close[i] < s1_4h_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20_4h_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to pivot or trend reversal
            if position == 1:
                # Exit on return to S1 or trend reversal
                if (close[i] <= s1_4h_aligned[i] or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on return to R1 or trend reversal
                if (close[i] >= r1_4h_aligned[i] or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_4hVolSpike"
timeframe = "4h"
leverage = 1.0