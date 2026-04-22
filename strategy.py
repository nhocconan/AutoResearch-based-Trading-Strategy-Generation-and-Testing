#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with volume confirmation and 1d EMA34 trend filter.
# Bollinger Band width measures volatility contraction (squeeze). Breakout after squeeze with volume
# spike and trend alignment captures explosive moves. Designed for low trade frequency (~20-40/year)
# to minimize fee decay. Works in both bull and bear markets by following higher timeframe trend
# and requiring volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA34 trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA on 1d close for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe (waits for 1d bar to close)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Bollinger Bands on 4h data (20-period, 2 std dev)
    close = prices['close'].values
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # Bollinger Band squeeze: width below 20-period average
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(sma_20[i]) or 
            np.isnan(std_20[i]) or 
            np.isnan(bb_width_ma[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        sma_val = sma_20[i]
        upper_val = upper_bb[i]
        lower_val = lower_bb[i]
        ema_val = ema_34_aligned[i]
        squeeze_val = squeeze[i]
        
        # Volume filter: current volume > 2.0 * 20-period average (strict filter for low frequency)
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: Bollinger Band breakout above upper band + squeeze + uptrend + volume spike
            if price > upper_val and squeeze_val and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: Bollinger Band breakout below lower band + squeeze + downtrend + volume spike
            elif price < lower_val and squeeze_val and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price breaks below lower Bollinger Band or trend breaks
                if price < lower_val or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price breaks above upper Bollinger Band or trend breaks
                if price > upper_val or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_BollingerSqueeze_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0