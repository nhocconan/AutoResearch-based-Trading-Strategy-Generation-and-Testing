#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long when price breaks above R4 with volume > 1.5x 20-bar avg AND 1w close > 1w EMA50
# - Short when price breaks below S4 with volume > 1.5x 20-bar avg AND 1w close < 1w EMA50
# - Exit when price returns to 1d pivot point (PP)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~20 trades/year (80 total over 4 years) to avoid fee drag
# - 1w trend filter ensures we only trade with the higher timeframe trend
# - Volume confirmation ensures institutional participation in breakouts

name = "6h_1w_camarilla_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 1d Camarilla levels (using previous day's OHLC)
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # Shift by 1 to use previous day's data (no look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # First value invalid
    
    # Calculate Camarilla levels
    rang = prev_high - prev_low
    pp = (prev_high + prev_low + prev_close) / 3
    r4 = pp + rang * 1.1 / 2
    s4 = pp - rang * 1.1 / 2
    
    # Pre-compute 6h volume confirmation: > 1.5x 20-bar average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r4[i]) or np.isnan(s4[i]) or 
            np.isnan(pp[i]) or np.isnan(volume_20_avg[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above R4 with volume spike and 1w uptrend
            if (prices['close'].iloc[i] > r4[i] and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema_50_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: price breaks below S4 with volume spike and 1w downtrend
            elif (prices['close'].iloc[i] < s4[i] and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema_50_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit long when price returns to pivot point
            if position == 1 and prices['close'].iloc[i] < pp[i]:
                position = 0
                signals[i] = 0.0
            # Exit short when price returns to pivot point
            elif position == -1 and prices['close'].iloc[i] > pp[i]:
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals