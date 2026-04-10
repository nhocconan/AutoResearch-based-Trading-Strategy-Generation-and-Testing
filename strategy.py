#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d trend filter
# - Long when price breaks above 20-period high with volume > 2.0x 20-bar average AND 1d close > 1d EMA50
# - Short when price breaks below 20-period low with volume > 2.0x 20-bar average AND 1d close < 1d EMA50
# - Exit when price retreats to midpoint of Donchian channel OR volume drops below 0.8x average
# - Uses 1d trend filter to avoid counter-trend trades in bear markets (2025+)
# - Higher volume threshold (2.0x) reduces trade frequency (target: 20-40 trades/year)
# - Focus on BTC/ETH; SOL-only strategies are low value and will be discarded

name = "4h_1d_donchian_breakout_volume_trend_v7"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute volume confirmation: > 2.0x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (2.0 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.8x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.8 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute aligned 1d data properly
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Align them to 4h timeframe
    h_1d_aligned = align_htf_to_ltf(prices, df_1d, h_1d)
    l_1d_aligned = align_htf_to_ltf(prices, df_1d, l_1d)
    c_1d_aligned = align_htf_to_ltf(prices, df_1d, c_1d)
    
    # Pre-compute 1d EMA(50) for trend filter
    ema50_1d = pd.Series(c_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(h_1d_aligned[i]) or np.isnan(l_1d_aligned[i]) or np.isnan(c_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get previous completed 1d bar values (1d data updates every 6 bars in 4h timeframe)
        if i >= 6:  # Need at least 6 4h bars to get previous 1d bar's data
            # Get index of previous completed 1d bar (look back to previous multiple of 6)
            lookback_idx = i - ((i % 6) + 6)  # Go back to start of previous 1d bar
            if lookback_idx >= 0:
                ph = h_1d_aligned[lookback_idx]
                pl = l_1d_aligned[lookback_idx]
                pc = c_1d_aligned[lookback_idx]
                
                # Calculate Donchian levels
                range_val = ph - pl
                if range_val > 0:
                    donchian_high = ph  # 20-period high
                    donchian_low = pl   # 20-period low
                    donchian_mid = (ph + pl) / 2  # Midpoint for exit
                    
                    if position == 0:  # Flat - look for new breakout entries
                        # Long breakout: price > Donchian high with volume spike AND 1d uptrend
                        if (prices['close'].iloc[i] > donchian_high and 
                            vol_spike.iloc[i] and 
                            prices['close'].iloc[i] > ema50_1d_aligned[i]):
                            position = 1
                            signals[i] = 0.25
                        # Short breakdown: price < Donchian low with volume spike AND 1d downtrend
                        elif (prices['close'].iloc[i] < donchian_low and 
                              vol_spike.iloc[i] and 
                              prices['close'].iloc[i] < ema50_1d_aligned[i]):
                            position = -1
                            signals[i] = -0.25
                    else:  # Have position - look for exit
                        # Exit conditions:
                        # 1. Price retreats to Donchian midpoint
                        # 2. Volume drops below 0.8x average (loss of momentum)
                        if position == 1:  # Long position
                            if (prices['close'].iloc[i] < donchian_mid or 
                                vol_weak.iloc[i]):
                                position = 0
                                signals[i] = 0.0
                            else:
                                signals[i] = 0.25  # Hold long
                        elif position == -1:  # Short position
                            if (prices['close'].iloc[i] > donchian_mid or 
                                vol_weak.iloc[i]):
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