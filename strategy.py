#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation
# - Long when price breaks above Donchian(20) high AND 1d EMA200 rising AND volume > 1.5x 20-bar avg
# - Short when price breaks below Donchian(20) low AND 1d EMA200 falling AND volume > 1.5x 20-bar avg
# - Exit when price crosses opposite Donchian level (reversal) or ATR-based stoploss
# - Uses 1d EMA200 for trend filter to align with higher timeframe bias
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 30-50 trades/year on 4h timeframe (120-200 total over 4 years)
# - Donchian channels work well in both trending and ranging markets with proper filters

name = "4h_1d_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute Donchian channels from 4h data (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Calculate Donchian levels: highest high and lowest low over 20 periods
    highest_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels (no additional delay needed for breakout)
    highest_high_aligned = align_htf_to_ltf(prices, prices, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, prices, lowest_low)
    
    # Pre-compute 1d EMA(200) for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute ATR(14) for dynamic stoploss
    tr1 = prices['high'] - prices['low']
    tr2 = abs(prices['high'] - prices['close'].shift(1))
    tr3 = abs(prices['low'] - prices['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(atr[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian high AND 1d uptrend with volume spike
            if (prices['close'].iloc[i] > highest_high_aligned[i] and 
                prices['close'].iloc[i] > ema200_1d_aligned[i] and  # price above 1d EMA200
                vol_spike.iloc[i]):
                position = 1
                entry_price = prices['close'].iloc[i]
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND 1d downtrend with volume spike
            elif (prices['close'].iloc[i] < lowest_low_aligned[i] and 
                  prices['close'].iloc[i] < ema200_1d_aligned[i] and  # price below 1d EMA200
                  vol_spike.iloc[i]):
                position = -1
                entry_price = prices['close'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit conditions
            exit_signal = False
            
            # Exit conditions:
            # 1. Price crosses opposite Donchian level (reversal signal)
            # 2. ATR-based stoploss (2.5 * ATR from entry)
            if position == 1:  # Long position
                # Exit on reversal: price breaks below Donchian low
                if prices['close'].iloc[i] < lowest_low_aligned[i]:
                    exit_signal = True
                # Exit on stoploss: price drops 2.5 * ATR below entry
                elif prices['close'].iloc[i] < entry_price - 2.5 * atr[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                # Exit on reversal: price breaks above Donchian high
                if prices['close'].iloc[i] > highest_high_aligned[i]:
                    exit_signal = True
                # Exit on stoploss: price rises 2.5 * ATR above entry
                elif prices['close'].iloc[i] > entry_price + 2.5 * atr[i]:
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