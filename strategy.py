#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 1-day Bollinger Band squeeze breakout with volume confirmation and ATR-based exit
# Bollinger Band squeeze indicates low volatility and potential breakout
# Breakout above upper band or below lower band with volume > 1.5x average confirms institutional participation
# ATR-based exit provides adaptive risk management
# Works in bull/bear as Bollinger Bands adapt to volatility
# Target: 20-30 trades/year per symbol (80-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Bollinger Bands on 1d close (20 period, 2 std dev)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    # Align Bollinger Bands to 4h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # 4h ATR for volatility-based exit (14 period)
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume confirmation: 1.5x average volume (20 period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 20, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Bollinger Band squeeze: bandwidth < 20th percentile of past 50 periods
        if i >= 50:
            bb_width = upper_bb_aligned[i] - lower_bb_aligned[i]
            past_widths = []
            for j in range(max(start, i-50), i):
                if not np.isnan(upper_bb_aligned[j]) and not np.isnan(lower_bb_aligned[j]):
                    past_widths.append(upper_bb_aligned[j] - lower_bb_aligned[j])
            if len(past_widths) >= 10:
                width_threshold = np.percentile(past_widths, 20)
                squeeze = bb_width < width_threshold
            else:
                squeeze = False
        else:
            squeeze = False
        
        if position == 0:
            # Enter long: price breaks above upper BB + squeeze + volume confirmation
            if (close[i] > upper_bb_aligned[i] and 
                squeeze and 
                volume[i] > 1.5 * vol_ma[i]):
                position = 1
                signals[i] = position_size
            # Enter short: price breaks below lower BB + squeeze + volume confirmation
            elif (close[i] < lower_bb_aligned[i] and 
                  squeeze and 
                  volume[i] > 1.5 * vol_ma[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle band or ATR-based stop
            middle_bb = sma_20_aligned[i] if 'sma_20_aligned' in locals() else (upper_bb_aligned[i] + lower_bb_aligned[i]) / 2
            if close[i] < middle_bb or close[i] < (entry_price - 1.5 * atr[i]) if 'entry_price' in locals() else False:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle band or ATR-based stop
            middle_bb = sma_20_aligned[i] if 'sma_20_aligned' in locals() else (upper_bb_aligned[i] + lower_bb_aligned[i]) / 2
            if close[i] > middle_bb or close[i] > (entry_price + 1.5 * atr[i]) if 'entry_price' in locals() else False:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Bollinger_Squeeze_Breakout_Volume_v1"
timeframe = "4h"
leverage = 1.0