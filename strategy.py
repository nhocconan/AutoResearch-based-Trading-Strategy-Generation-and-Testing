#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level with volume > 1.5x 20-bar average AND 4h close > 4h EMA50
# - Short when price breaks below Camarilla L3 level with volume > 1.5x 20-bar average AND 4h close < 4h EMA50
# - Exit when price retreats to Camarilla H4/L4 levels OR volume drops below 0.7x average
# - Session filter: only trade 08-20 UTC to avoid low-liquidity hours
# - Uses 4h trend filter to avoid counter-trend trades in bear markets (2025+)
# - Moderate volume threshold (1.5x) balances signal quality and trade frequency (target: 15-35 trades/year)
# - Focus on BTC/ETH; SOL-only strategies are low value and will be discarded

name = "1h_4h_camarilla_breakout_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.7x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.7 * volume_20_avg)
    
    # Pre-compute aligned 4h data properly
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    # Align them to 1h timeframe
    h_4h_aligned = align_htf_to_ltf(prices, df_4h, h_4h)
    l_4h_aligned = align_htf_to_ltf(prices, df_4h, l_4h)
    c_4h_aligned = align_htf_to_ltf(prices, df_4h, c_4h)
    
    # Pre-compute 4h EMA(50) for trend filter
    ema50_4h = pd.Series(c_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if outside trading session or any required data is invalid
        if not in_session[i] or (np.isnan(ema50_4h_aligned[i]) or np.isnan(volume_20_avg[i]) or 
                                 np.isnan(h_4h_aligned[i]) or np.isnan(l_4h_aligned[i]) or np.isnan(c_4h_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Get previous completed 4h bar values
        # Since 1h timeframe, there are 4 bars per 4h bar
        if i >= 4:  # Need at least 4 1h bars to get previous 4h bar's data
            # Get index of previous completed 4h bar (look back 4 bars)
            prev_4h_idx = i - 4
            
            if prev_4h_idx >= 0:
                ph = h_4h_aligned[prev_4h_idx]  # Previous 4h bar's high
                pl = l_4h_aligned[prev_4h_idx]  # Previous 4h bar's low
                pc = c_4h_aligned[prev_4h_idx]  # Previous 4h bar's close
                
                # Calculate Camarilla levels
                range_val = ph - pl
                if range_val > 0:
                    camarilla_h3 = pc + (range_val * 1.1 / 4)
                    camarilla_l3 = pc - (range_val * 1.1 / 4)
                    camarilla_h4 = pc + (range_val * 1.1 / 2)
                    camarilla_l4 = pc - (range_val * 1.1 / 2)
                    
                    if position == 0:  # Flat - look for new breakout entries
                        # Long breakout: price > Camarilla H3 with volume spike AND 4h uptrend
                        if (prices['close'].iloc[i] > camarilla_h3 and 
                            vol_spike.iloc[i] and 
                            prices['close'].iloc[i] > ema50_4h_aligned[i]):
                            position = 1
                            signals[i] = 0.20
                        # Short breakdown: price < Camarilla L3 with volume spike AND 4h downtrend
                        elif (prices['close'].iloc[i] < camarilla_l3 and 
                              vol_spike.iloc[i] and 
                              prices['close'].iloc[i] < ema50_4h_aligned[i]):
                            position = -1
                            signals[i] = -0.20
                    else:  # Have position - look for exit
                        # Exit conditions:
                        # 1. Price retreats to Camarilla H4/L4 levels
                        # 2. Volume drops below 0.7x average (loss of momentum)
                        if position == 1:  # Long position
                            if (prices['close'].iloc[i] < camarilla_h4 or 
                                vol_weak.iloc[i]):
                                position = 0
                                signals[i] = 0.0
                            else:
                                signals[i] = 0.20  # Hold long
                        elif position == -1:  # Short position
                            if (prices['close'].iloc[i] > camarilla_l4 or 
                                vol_weak.iloc[i]):
                                position = 0
                                signals[i] = 0.0
                            else:
                                signals[i] = -0.20  # Hold short
                else:
                    # Hold current position
                    if position == 0:
                        signals[i] = 0.0
                    elif position == 1:
                        signals[i] = 0.20
                    else:
                        signals[i] = -0.20
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
        else:
            # Not enough data yet, hold flat
            signals[i] = 0.0
    
    return signals