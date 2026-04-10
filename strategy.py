#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA trend filter and volume confirmation
# - Long when price breaks above 20-period high AND 12h EMA50 rising AND volume > 1.5x 20-bar avg
# - Short when price breaks below 20-period low AND 12h EMA50 falling AND volume > 1.5x 20-bar avg
# - Exit with ATR-based trailing stop: signal=0 when long and price < highest_high - 2.5*ATR(14) OR short and price < lowest_low + 2.5*ATR(14)
# - Uses 12h EMA50 for trend filter to avoid counter-trend trades
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 25-35 trades/year on 4h timeframe (100-140 total over 4 years)
# - Donchian breakouts capture momentum; trend filter improves win rate in bear markets

name = "4h_12h_donchian_breakout_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Pre-compute ATR(14) for trailing stop
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0  # Track highest high since long entry
    lowest_since_entry = 0.0   # Track lowest low since short entry
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above 20-period high AND 12h uptrend with volume spike
            if (prices['close'].iloc[i] > highest_high[i] and 
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and  # 12h EMA50 rising
                vol_spike.iloc[i]):
                position = 1
                highest_since_entry = prices['high'].iloc[i]
                signals[i] = 0.25
            # Short when price breaks below 20-period low AND 12h downtrend with volume spike
            elif (prices['close'].iloc[i] < lowest_low[i] and 
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and  # 12h EMA50 falling
                  vol_spike.iloc[i]):
                position = -1
                lowest_since_entry = prices['low'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for ATR trailing stop exit
            # Update highest/lowest since entry
            if position == 1:
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                # Exit long when price drops below highest_high - 2.5*ATR
                if prices['close'].iloc[i] < highest_since_entry - 2.5 * atr[i]:
                    position = 0
                    highest_since_entry = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # Exit short when price rises above lowest_low + 2.5*ATR
                if prices['close'].iloc[i] > lowest_since_entry + 2.5 * atr[i]:
                    position = 0
                    lowest_since_entry = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals