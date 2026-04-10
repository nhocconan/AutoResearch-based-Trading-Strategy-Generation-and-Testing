#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level with volume > 1.5x average AND weekly close > weekly EMA50
# - Short when price breaks below Camarilla L3 level with volume > 1.5x average AND weekly close < weekly EMA50
# - Exit when price retreats to Camarilla H4/L4 levels or volume drops below average
# - Weekly trend filter ensures alignment with major trend across market cycles
# - Volume confirmation prevents false breakouts
# - Targets 7-25 trades/year (30-100 total over 4 years) to avoid fee drag
# - Daily timeframe avoids excessive trading while capturing multi-day moves
# - Combines Camarilla structure with weekly trend/volume filters for high-probability breakouts

name = "1d_1w_camarilla_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < average volume for exit
    vol_normal = prices['volume'] < volume_20_avg
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Get previous completed 1d bar values (shifted by 1 to avoid look-ahead)
        if i >= 1:
            ph = prices['high'].iloc[i-1]  # Previous day's high
            pl = prices['low'].iloc[i-1]   # Previous day's low
            pc = prices['close'].iloc[i-1] # Previous day's close
            
            if not (np.isnan(ph) or np.isnan(pl) or np.isnan(pc)):
                # Calculate Camarilla levels
                range_val = ph - pl
                if range_val > 0:
                    camarilla_h3 = pc + (range_val * 1.1 / 4)
                    camarilla_l3 = pc - (range_val * 1.1 / 4)
                    camarilla_h4 = pc + (range_val * 1.1 / 2)
                    camarilla_l4 = pc - (range_val * 1.1 / 2)
                    
                    if position == 0:  # Flat - look for new breakout entries
                        # Long breakout: price > Camarilla H3 with volume spike AND weekly uptrend
                        if (prices['close'].iloc[i] > camarilla_h3 and 
                            vol_spike.iloc[i] and 
                            prices['close'].iloc[i] > ema50_1w_aligned[i]):
                            position = 1
                            signals[i] = 0.25
                        # Short breakdown: price < Camarilla L3 with volume spike AND weekly downtrend
                        elif (prices['close'].iloc[i] < camarilla_l3 and 
                              vol_spike.iloc[i] and 
                              prices['close'].iloc[i] < ema50_1w_aligned[i]):
                            position = -1
                            signals[i] = -0.25
                    else:  # Have position - look for exit
                        # Exit conditions:
                        # 1. Price retreats to Camarilla H4/L4 levels
                        # 2. Volume drops below average (loss of momentum)
                        if position == 1:  # Long position
                            if (prices['close'].iloc[i] < camarilla_h4 or 
                                vol_normal.iloc[i]):
                                position = 0
                                signals[i] = 0.0
                            else:
                                signals[i] = 0.25  # Hold long
                        elif position == -1:  # Short position
                            if (prices['close'].iloc[i] > camarilla_l4 or 
                                vol_normal.iloc[i]):
                                position = 0
                                signals[i] = 0.0
                            else:
                                signals[i] = -0.25  # Hold short
                else:
                    signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            else:
                signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
        else:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
    
    return signals