#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation.
# In low volatility regimes (Bollinger Band width at 20-period low), price often breaks out strongly.
# We combine this with 1d EMA34 trend filter to ensure we trade in the direction of higher timeframe trend,
# and require volume > 1.5x 20-period average to confirm institutional participation.
# Designed for low trade frequency (~20-40/year) to minimize fee decay.
# Works in both bull and bear markets by following 1d trend direction.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for Bollinger Bands and EMA (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Bollinger Bands on 1d close
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2.0 * std_20
    lower_bb = sma_20 - 2.0 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # Calculate 34-period EMA on 1d close for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 4h timeframe (waits for 1d bar to close)
    sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period average volume for volume confirmation (on 4h data)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(sma_20_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or 
            np.isnan(bb_width_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        sma = sma_20_aligned[i]
        upper = upper_bb_aligned[i]
        lower = lower_bb_aligned[i]
        width = bb_width_aligned[i]
        ema = ema_34_aligned[i]
        
        # Bollinger Band squeeze: width at 20-period low (bottom 20%)
        if i >= 20:
            width_history = bb_width_aligned[max(0, i-19):i+1]
            width_rank = np.sum(width_history <= width) / len(width_history) * 100
            squeeze = width_rank <= 20  # Bottom 20% = squeeze
        else:
            squeeze = False
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_confirm = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: squeeze breakout above upper BB + uptrend + volume
            if squeeze and price > upper and price > ema and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: squeeze breakout below lower BB + downtrend + volume
            elif squeeze and price < lower and price < ema and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price breaks below middle BB or trend breaks
                if price < sma or price < ema:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price breaks above middle BB or trend breaks
                if price > sma or price > ema:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_BollingerSqueeze_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0