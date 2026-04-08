#!/usr/bin/env python3
# 1d_keltner_breakout_1w_trend_volume_v1
# Hypothesis: Keltner Channel breakout with weekly trend and volume confirmation on daily timeframe.
# Long when price closes above upper Keltner band (EMA10 + 2*ATR) and price > weekly EMA200 with volume > 1.3x average.
# Short when price closes below lower Keltner band (EMA10 - 2*ATR) and price < weekly EMA200 with volume > 1.3x average.
# Exit on opposite Keltner band touch or when volume drops below average.
# Designed to capture strong trends with volume confirmation to reduce whipsaw in both bull and bear markets.
# Target: 30-100 total trades over 4 years (~7-25/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_keltner_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Keltner Channel components (10-period EMA, 10-period ATR)
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    tr1 = np.maximum(high - low, np.abs(high - np.concatenate([[close[0]], close[:-1]])))
    tr2 = np.maximum(np.abs(low - np.concatenate([[close[0]], close[:-1]])), tr1)
    tr = tr2
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    kc_upper = ema_10 + 2 * atr
    kc_lower = ema_10 - 2 * atr
    
    # Average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(ema_200_1w_aligned[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches lower Keltner band or volume drops below average
            if close[i] <= kc_lower[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches upper Keltner band or volume drops below average
            if close[i] >= kc_upper[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.3x average volume
            volume_ok = volume[i] > 1.3 * avg_volume[i]
            
            # Trend filter: price vs weekly EMA200
            price_above_ema = close[i] > ema_200_1w_aligned[i]
            price_below_ema = close[i] < ema_200_1w_aligned[i]
            
            # Keltner breakout entries
            if close[i] > kc_upper[i] and price_above_ema and volume_ok:
                # Additional confirmation: previous close was at or below upper band to confirm breakout
                if i > 0 and close[i-1] <= kc_upper[i-1]:
                    position = 1
                    signals[i] = 0.25
            elif close[i] < kc_lower[i] and price_below_ema and volume_ok:
                # Additional confirmation: previous close was at or above lower band to confirm breakdown
                if i > 0 and close[i-1] >= kc_lower[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals