#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Long when Williams %R(14) < -80 (oversold) AND price > 1d EMA200 (bullish bias) AND volume > 1.5x average
# - Short when Williams %R(14) > -20 (overbought) AND price < 1d EMA200 (bearish bias) AND volume > 1.5x average
# - Exit when Williams %R returns to -50 level OR volume drops below 0.7x average (loss of momentum)
# - Uses 1d EMA200 for trend filter to avoid counter-trend trades
# - Williams %R is effective in ranging markets (2025+) and captures mean reversion
# - Volume confirmation reduces false signals
# - Target: 15-25 trades/year (12h timeframe, tight entries)

name = "12h_1d_williamsr_meanreversion_volume_trend_v4"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Williams %R on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          ((highest_high - close_1d) / (highest_high - lowest_low)) * -100, 
                          -50)
    
    # Align Williams %R to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Pre-compute 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.7x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.7 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get previous completed 1d bar values (need to ensure we use completed bar)
        # For 12h timeframe, 1d data updates every 2 bars
        if i >= 2:  # Need at least 2 bars to look back
            # Determine index of previous completed 1d bar
            # Each 1d bar spans 2 consecutive 12h bars
            if i % 2 == 0:  # Even index = start of new 1d bar
                lookback_idx = i - 2  # Previous completed 1d bar
            else:  # Odd index = second half of 1d bar
                lookback_idx = i - 1  # Previous completed 1d bar
            
            if lookback_idx >= 0:
                wr = williams_r_aligned[lookback_idx]
                ema200 = ema200_1d_aligned[lookback_idx]
                close_price = prices['close'].iloc[i]
                
                if position == 0:  # Flat - look for new mean reversion entries
                    # Long when oversold AND bullish trend bias AND volume confirmation
                    if (wr < -80 and 
                        close_price > ema200 and 
                        vol_spike.iloc[i]):
                        position = 1
                        signals[i] = 0.25
                    # Short when overbought AND bearish trend bias AND volume confirmation
                    elif (wr > -20 and 
                          close_price < ema200 and 
                          vol_spike.iloc[i]):
                        position = -1
                        signals[i] = -0.25
                else:  # Have position - look for exit
                    # Exit when Williams %R returns to mean (-50) OR loss of momentum
                    if position == 1:  # Long position
                        if (wr >= -50 or vol_weak.iloc[i]):
                            position = 0
                            signals[i] = 0.0
                        else:
                            signals[i] = 0.25  # Hold long
                    elif position == -1:  # Short position
                        if (wr <= -50 or vol_weak.iloc[i]):
                            position = 0
                            signals[i] = 0.0
                        else:
                            signals[i] = -0.25  # Hold short
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Not enough data yet, hold flat
            signals[i] = 0.0
    
    return signals