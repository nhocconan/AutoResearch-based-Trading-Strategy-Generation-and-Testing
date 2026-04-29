#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation (>1.8x 20-period average)
# Camarilla pivots provide precise support/resistance levels effective in ranging and trending markets
# 1d EMA34 ensures alignment with higher timeframe trend to avoid counter-trend trades
# Volume confirmation (>1.8x) filters weak breakouts and ensures institutional participation
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe

name = "12h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for 1d EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        
        # Calculate Camarilla levels for previous 1d bar (use prior completed day)
        if i >= 1:
            # Get previous completed 1d bar data
            prev_day_idx = len(df_1d) - 1  # align_htf_to_ltf ensures proper alignment
            # Simpler approach: use rolling window on 1d data aligned to 12h
            # We'll calculate Camarilla from 12h data using 1d lookback
            lookback = min(i // 2, 100)  # approximate 1d lookback in 12h bars (max 100)
            if lookback >= 20:
                window_high = np.max(high[max(0, i-lookback):i])
                window_low = np.min(low[max(0, i-lookback):i])
                window_close = close[i-1]  # previous bar close
                
                # Camarilla equations
                rang = window_high - window_low
                if rang > 0:
                    R3 = window_close + rang * 1.1 / 4
                    S3 = window_close - rang * 1.1 / 4
                    
                    # Volume confirmation: current volume > 1.8x 20-period average (tight threshold)
                    if i >= 20:
                        vol_ma_20 = np.mean(volume[max(0, i-20):i])
                        vol_confirm = curr_volume > 1.8 * vol_ma_20
                        
                        # Handle exits
                        if position == 1:  # Long position
                            # Exit: price closes below S3 OR price closes below 1d EMA34
                            if curr_close < S3 or curr_close < curr_ema_1d:
                                signals[i] = 0.0
                                position = 0
                            else:
                                signals[i] = 0.25
                                
                        elif position == -1:  # Short position
                            # Exit: price closes above R3 OR price closes above 1d EMA34
                            if curr_close > R3 or curr_close > curr_ema_1d:
                                signals[i] = 0.0
                                position = 0
                            else:
                                signals[i] = -0.25
                                
                        else:  # Flat - look for new entries
                            # Long entry: price breaks above R3 + price above 1d EMA34 + volume confirmation
                            if (curr_close > R3 and 
                                curr_close > curr_ema_1d and 
                                vol_confirm):
                                signals[i] = 0.25
                                position = 1
                            # Short entry: price breaks below S3 + price below 1d EMA34 + volume confirmation
                            elif (curr_close < S3 and 
                                  curr_close < curr_ema_1d and 
                                  vol_confirm):
                                signals[i] = -0.25
                                position = -1
                            else:
                                signals[i] = 0.0
                    else:
                        signals[i] = 0.0
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals