#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w trend filter (1w EMA200) and volume confirmation
# - Long when price breaks above 12h Donchian upper band AND 1w EMA200 rising AND volume > 2.0x 20-bar avg
# - Short when price breaks below 12h Donchian lower band AND 1w EMA200 falling AND volume > 2.0x 20-bar avg
# - Exit with ATR(14) trailing stop: long exits when price < highest_high - 2.5*ATR, short exits when price > lowest_low + 2.5*ATR
# - Uses 1w EMA200 for strong trend filter to avoid counter-trend trades in bear markets
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-25 trades/year on 12h timeframe (50-100 total over 4 years)
# - Donchian breakouts capture strong moves; 1w EMA200 filter ensures we only trade with the primary trend
# - Volume confirmation reduces false breakouts
# - ATR trailing stop lets winners run while controlling risk

name = "12h_1w_donchian_breakout_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w EMA(200) for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Pre-compute 12h Donchian channels (20-period)
    high_12h = prices['high'].rolling(window=20, min_periods=20).max().values
    low_12h = prices['low'].rolling(window=20, min_periods=20).min().values
    donchian_high = high_12h
    donchian_low = low_12h
    
    # Pre-compute ATR(14) for trailing stop
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift(1))
    low_close = np.abs(prices['low'] - prices['close'].shift(1))
    tr = np.maximum(np.maximum(high_low.values, high_close.values), low_close.values)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute volume confirmation: > 2.0x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (2.0 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    # For trailing stop: track highest high since long entry, lowest low since short entry
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(100, n):  # Start after warmup (need 200 for 1w EMA, 20 for Donchian, 14 for ATR)
        # Skip if any required data is invalid
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(atr[i]) or 
            np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            # Update tracking variables
            if position == 1:
                highest_since_entry[i] = max(highest_since_entry[i-1], prices['high'].iloc[i]) if i > 0 else prices['high'].iloc[i]
                lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else 0
            elif position == -1:
                highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else 0
                lowest_since_entry[i] = min(lowest_since_entry[i-1], prices['low'].iloc[i]) if i > 0 else prices['low'].iloc[i]
            else:
                highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else 0
                lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else 0
            continue
        
        # Update tracking variables for trailing stop
        if position == 1:
            highest_since_entry[i] = max(highest_since_entry[i-1], prices['high'].iloc[i]) if i > 0 else prices['high'].iloc[i]
            lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else 0
        elif position == -1:
            highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else 0
            lowest_since_entry[i] = min(lowest_since_entry[i-1], prices['low'].iloc[i]) if i > 0 else prices['low'].iloc[i]
        else:
            highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else 0
            lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else 0
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian upper band AND 1w uptrend with volume spike
            if (prices['close'].iloc[i] > donchian_high[i] and 
                prices['close'].iloc[i] > ema200_1w_aligned[i] and  # price above 1w EMA200
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian lower band AND 1w downtrend with volume spike
            elif (prices['close'].iloc[i] < donchian_low[i] and 
                  prices['close'].iloc[i] < ema200_1w_aligned[i] and  # price below 1w EMA200
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for ATR trailing stop exit
            exit_signal = False
            if position == 1:  # Long position
                # Exit when price drops below highest high since entry minus 2.5*ATR
                if prices['close'].iloc[i] < highest_since_entry[i] - 2.5 * atr[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                # Exit when price rises above lowest low since entry plus 2.5*ATR
                if prices['close'].iloc[i] > lowest_since_entry[i] + 2.5 * atr[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals