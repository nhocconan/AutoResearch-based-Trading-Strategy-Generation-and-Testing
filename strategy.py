#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level with volume > 1.3x average AND 4h close > 4h EMA50
# - Short when price breaks below Camarilla L3 level with volume > 1.3x average AND 4h close < 4h EMA50
# - Exit when price retreats to Camarilla H4/L4 levels or volume drops below average
# - 4h trend filter ensures alignment with higher timeframe trend
# - Volume confirmation prevents false breakouts
# - Session filter (08-20 UTC) reduces noise trades
# - Targets 15-37 trades/year (60-150 total over 4 years) to avoid fee drag
# - Uses discrete position size of 0.20 to minimize churn

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
    
    # Pre-compute 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Pre-compute volume confirmation: > 1.3x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.3 * volume_20_avg)
    
    # Pre-compute volume filter: < average volume for exit
    vol_normal = prices['volume'] < volume_20_avg
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if outside trading session or any required data is invalid
        if not session_filter.iloc[i] or np.isnan(ema50_4h_aligned[i]) or np.isnan(volume_20_avg[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Calculate Camarilla levels from previous 4h bar
        # Create aligned arrays for 4h OHLC
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        c_4h = df_4h['close'].values
        
        h_4h_aligned = align_htf_to_ltf(prices, df_4h, h_4h)
        l_4h_aligned = align_htf_to_ltf(prices, df_4h, l_4h)
        c_4h_aligned = align_htf_to_ltf(prices, df_4h, c_4h)
        
        # Get previous completed 4h bar values (shifted by 1 to avoid look-ahead)
        if i >= 1:
            ph = h_4h_aligned[i-1]  # Previous 4h bar's high
            pl = l_4h_aligned[i-1]  # Previous 4h bar's low
            pc = c_4h_aligned[i-1]  # Previous 4h bar's close
            
            if not (np.isnan(ph) or np.isnan(pl) or np.isnan(pc)):
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
                        # 2. Volume drops below average (loss of momentum)
                        if position == 1:  # Long position
                            if (prices['close'].iloc[i] < camarilla_h4 or 
                                vol_normal.iloc[i]):
                                position = 0
                                signals[i] = 0.0
                            else:
                                signals[i] = 0.20  # Hold long
                        elif position == -1:  # Short position
                            if (prices['close'].iloc[i] > camarilla_l4 or 
                                vol_normal.iloc[i]):
                                position = 0
                                signals[i] = 0.0
                            else:
                                signals[i] = -0.20  # Hold short
                else:
                    signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            else:
                signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
        else:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
    
    return signals