#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA trend filter and ATR-based exits
# - Uses 12h EMA(50) for trend direction (bullish when price > EMA, bearish when price < EMA)
# - Uses 6h Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# - Long when Bull Power > 0 and Bear Power < 0 (strong bullish momentum)
# - Short when Bear Power < 0 and Bull Power < 0 (strong bearish momentum)
# - ATR(14) trailing stop: exit long when price drops 2.5*ATR from highest high since entry
# - Exit short when price rises 2.5*ATR from lowest low since entry
# - Position size: 0.25 (25% of capital) to manage drawdown in bear markets
# - Works in bull via trend continuation, in bear via mean reversion at extremes
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years)

name = "6h_12h_elder_ray_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 6h Elder Ray components: EMA(13) of close for power calculation
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA(13)
    bear_power = low - ema_13   # Bear Power = Low - EMA(13)
    
    # Pre-compute 6h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(atr[i]) or atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 12h EMA
        uptrend = close[i] > ema_12h_aligned[i]
        downtrend = close[i] < ema_12h_aligned[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # Exit conditions: ATR trailing stop
            if close[i] < highest_high_since_entry - 2.5 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # Exit conditions: ATR trailing stop
            if close[i] > lowest_low_since_entry + 2.5 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for entries based on Elder Ray power and 12h trend
            # Long: strong bullish momentum (Bull Power > 0) AND bearish weakness (Bear Power < 0) in uptrend
            if uptrend and bull_power[i] > 0 and bear_power[i] < 0:
                position = 1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = 0.25
            # Short: strong bearish momentum (Bear Power < 0) AND bullish weakness (Bull Power < 0) in downtrend
            elif downtrend and bear_power[i] < 0 and bull_power[i] < 0:
                position = -1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = -0.25
    
    return signals