#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with volume confirmation and 4h/1d trend filter
# - Uses 4h HTF for trend direction (price above/below 20 EMA)
# - Uses 1d HTF for Camarilla pivot levels (H3, L3 from prior day)
# - 1h timeframe for entry timing: long when price closes above H3 with volume > 1.8x 20-period average AND 4h uptrend
# - Short when price closes below L3 with volume > 1.8x 20-period average AND 4h downtrend
# - Fixed position size 0.20 to control drawdown and reduce fee churn
# - Session filter: only trade 08-20 UTC to avoid low-volume periods
# - Target: 15-35 trades/year (60-140 total over 4 years) on 1h timeframe

name = "1h_4h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC) ONCE before loop
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 4h EMA(20) for trend
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate prior day's Camarilla H3 and L3 levels
    # H3 = close + 1.0 * (high - low)
    # L3 = close - 1.0 * (high - low)
    high_low_diff = high_1d - low_1d
    H3 = close_1d + 1.0 * high_low_diff
    L3 = close_1d - 1.0 * high_low_diff
    
    # Align Camarilla levels to 1h timeframe (wait for completed 1d bar)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Pre-compute volume confirmation (20-period average) ONCE before loop
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not in trading session or missing data
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.8x average
        volume_confirmed = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit if price closes below L3 (mean reversion) or trend breaks
            if close[i] < L3_aligned[i] or close[i] < ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit if price closes above H3 (mean reversion) or trend breaks
            if close[i] > H3_aligned[i] or close[i] > ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Entry logic: Camarilla breakout + volume confirmation + 4h trend filter
            if volume_confirmed:
                # Long entry: price closes above H3 AND 4h uptrend (price > EMA20)
                if close[i] > H3_aligned[i] and close[i] > ema_4h_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short entry: price closes below L3 AND 4h downtrend (price < EMA20)
                elif close[i] < L3_aligned[i] and close[i] < ema_4h_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals