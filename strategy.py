#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level with volume > 1.8x average AND 12h close > 12h EMA20
# - Short when price breaks below Camarilla L3 level with volume > 1.8x average AND 12h close < 12h EMA20
# - Exit when price retreats to Camarilla H4/L4 levels
# - Uses 12h trend filter to avoid counter-trend trades in ranging markets
# - Higher volume threshold (1.8x) reduces false breakouts and trade frequency
# - Targets 15-30 trades/year (60-120 total over 4 years) to avoid fee drag
# - Camarilla pivots work well in ranging markets; combined with 12h trend/volume filters for quality breakouts

name = "4h_12h_camarilla_breakout_volume_trend_v1"
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
    
    # Pre-compute 12h EMA(20) for trend filter
    close_12h = df_12h['close'].values
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Pre-compute volume confirmation: > 1.8x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.8 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema20_12h_aligned[i]) or np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Check if we have enough 12h data for Camarilla calculation (need previous completed 12h bar)
        # For 4h timeframe, there are 3 bars per 12h bar
        if i >= 3:
            # Get index of previous completed 12h bar
            prev_12h_idx = i - 3
            
            if prev_12h_idx >= 0 and not (np.isnan(df_12h['high'].values[min(prev_12h_idx//3, len(df_12h)-1)]) if prev_12h_idx//3 < len(df_12h) else True):
                # Get previous completed 12h bar OHLC using proper alignment
                # We need to access the 12h data directly for Camarilla calculation
                h_12h = df_12h['high'].values
                l_12h = df_12h['low'].values
                c_12h = df_12h['close'].values
                
                # Calculate which 12h bar we're in
                bar_12h_idx = i // 3
                if bar_12h_idx > 0:  # Need at least one previous 12h bar
                    prev_12h_bar = bar_12h_idx - 1
                    if prev_12h_bar < len(h_12h):
                        ph = h_12h[prev_12h_bar]  # Previous 12h bar's high
                        pl = l_12h[prev_12h_bar]  # Previous 12h bar's low
                        pc = c_12h[prev_12h_bar]  # Previous 12h bar's close
                        
                        # Calculate Camarilla levels
                        range_val = ph - pl
                        if range_val > 0:
                            camarilla_h3 = pc + (range_val * 1.1 / 4)
                            camarilla_l3 = pc - (range_val * 1.1 / 4)
                            camarilla_h4 = pc + (range_val * 1.1 / 2)
                            camarilla_l4 = pc - (range_val * 1.1 / 2)
                            
                            if position == 0:  # Flat - look for new breakout entries
                                # Long breakout: price > Camarilla H3 with volume spike AND 12h uptrend
                                if (prices['close'].iloc[i] > camarilla_h3 and 
                                    vol_spike.iloc[i] and 
                                    prices['close'].iloc[i] > ema20_12h_aligned[i]):
                                    position = 1
                                    signals[i] = 0.25
                                # Short breakdown: price < Camarilla L3 with volume spike AND 12h downtrend
                                elif (prices['close'].iloc[i] < camarilla_l3 and 
                                      vol_spike.iloc[i] and 
                                      prices['close'].iloc[i] < ema20_12h_aligned[i]):
                                    position = -1
                                    signals[i] = -0.25
                            else:  # Have position - look for exit
                                # Exit when price retreats to Camarilla H4/L4 levels
                                if position == 1:  # Long position
                                    if prices['close'].iloc[i] < camarilla_h4:
                                        position = 0
                                        signals[i] = 0.0
                                    else:
                                        signals[i] = 0.25  # Hold long
                                elif position == -1:  # Short position
                                    if prices['close'].iloc[i] > camarilla_l4:
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
                    # Not enough 12h bars yet, hold flat
                    signals[i] = 0.0
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