#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation
# - Long when price breaks above 20-day high AND weekly EMA50 rising AND volume > 2.0x 20-bar avg
# - Short when price breaks below 20-day low AND weekly EMA50 falling AND volume > 2.0x 20-bar avg
# - Exit with ATR-based trailing stop: signal→0 when price < highest_high - 2.5*ATR (long) or price > lowest_low + 2.5*ATR (short)
# - Uses weekly EMA50 for trend filter to avoid counter-trend trades and reduce whipsaw in bear markets
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 15-25 trades/year on 1d timeframe (60-100 total over 4 years)

name = "1d_1w_donchian_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute ATR(14) for trailing stop
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Pre-compute Donchian channels from daily data (20-period)
    highest_20 = pd.Series(close).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(close).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Pre-compute volume confirmation: > 2.0x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (2.0 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(atr[i]) or 
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
            # Long when price breaks above 20-day high AND weekly uptrend with volume spike
            if (prices['close'].iloc[i] > highest_20[i] and 
                prices['close'].iloc[i] > ema50_1w_aligned[i] and  # price above weekly EMA50
                vol_spike.iloc[i]):
                position = 1
                entry_price = prices['close'].iloc[i]
                highest_since_entry = entry_price
                lowest_since_entry = entry_price
                signals[i] = 0.25
            # Short when price breaks below 20-day low AND weekly downtrend with volume spike
            elif (prices['close'].iloc[i] < lowest_20[i] and 
                  prices['close'].iloc[i] < ema50_1w_aligned[i] and  # price below weekly EMA50
                  vol_spike.iloc[i]):
                position = -1
                entry_price = prices['close'].iloc[i]
                highest_since_entry = entry_price
                lowest_since_entry = entry_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - update extremes and check trailing stop
            # Update highest/lowest since entry
            if position == 1:
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # Check long trailing stop: exit if price drops 2.5*ATR from highest since entry
                if prices['close'].iloc[i] < highest_since_entry - 2.5 * atr[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # Check short trailing stop: exit if price rises 2.5*ATR from lowest since entry
                if prices['close'].iloc[i] > lowest_since_entry + 2.5 * atr[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals