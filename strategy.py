#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d RSI divergence + volume confirmation
# Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) measures bull/bear strength
# Divergence: price makes new high/low but Elder Ray does not confirms weakness
# Long when: Bear Power divergence (price new low, Bear Power higher) + RSI(14) < 30 + volume > 1.5x avg
# Short when: Bull Power divergence (price new high, Bull Power lower) + RSI(14) > 70 + volume > 1.5x avg
# Exit when Elder Ray crosses zero or RSI returns to neutral zone (40-60)
# Designed for 15-30 trades/year on 6h to capture exhaustion moves in BTC/ETH bull and bear markets

name = "6h_1d_elder_ray_divergence_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 13-period EMA for Elder Ray (using close)
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power_1d = high - ema13_1d  # High - EMA13
    bear_power_1d = low - ema13_1d   # Low - EMA13
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # 14-period RSI on 1d close
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.values
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # 20-period average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Arrays to track divergence conditions
    bull_div = np.full(n, False)  # Price new high but Bull Power lower (bearish divergence)
    bear_div = np.full(n, False)  # Price new low but Bear Power higher (bullish divergence)
    
    # Lookback period for divergence detection (10 periods)
    lookback = 10
    
    for i in range(lookback, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(vol_ma_20[i])):
            continue
            
        # Bullish divergence: price makes new low but Bear Power is higher (less negative)
        price_low_lookback = np.min(low[i-lookback:i+1])
        bear_power_low_lookback = np.min(bear_power_aligned[i-lookback:i+1])
        price_new_low = low[i] == price_low_lookback
        bear_power_higher = bear_power_aligned[i] > bear_power_aligned[i-1]
        bear_div[i] = price_new_low and bear_power_higher
        
        # Bearish divergence: price makes new high but Bull Power is lower (weaker)
        price_high_lookback = np.max(high[i-lookback:i+1])
        bull_power_high_lookback = np.max(bull_power_aligned[i-lookback:i+1])
        price_new_high = high[i] == price_high_lookback
        bull_power_lower = bull_power_aligned[i] < bull_power_aligned[i-1]
        bull_div[i] = price_new_high and bull_power_lower
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # RSI conditions
        rsi_oversold = rsi_aligned[i] < 30
        rsi_overbought = rsi_aligned[i] > 70
        rsi_neutral = (rsi_aligned[i] >= 40) & (rsi_aligned[i] <= 60)
        
        # Entry conditions
        long_entry = bear_div[i] and volume_filter and rsi_oversold
        short_entry = bull_div[i] and volume_filter and rsi_overbought
        
        # Exit conditions: Elder Ray crosses zero OR RSI returns to neutral
        long_exit = (bear_power_aligned[i] > 0) or rsi_neutral
        short_exit = (bull_power_aligned[i] < 0) or rsi_neutral
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals