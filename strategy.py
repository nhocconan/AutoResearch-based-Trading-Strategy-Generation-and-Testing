#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean-reversion strategy using 4h RSI extremes with 1d trend filter and volume confirmation.
# Buy when 4h RSI < 30 (oversold) and price > 1d EMA50 (uptrend) with volume spike (>1.5x 20-period average).
# Sell when 4h RSI > 70 (overbought) and price < 1d EMA50 (downtrend) with volume spike.
# Designed for low trade frequency (~15-35/year) to minimize fee drag. Works in both bull and bear markets
# by combining mean-reversion entries with trend filtering to avoid counter-trend trades.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for RSI calculation (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 14-period RSI on 4h close
    delta = pd.Series(close_4h).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h = rsi_4h.replace([np.inf, -np.inf], 100).fillna(100).values
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on 1d close for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h RSI and 1d EMA to 1h timeframe
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(rsi_4h_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        rsi_val = rsi_4h_aligned[i]
        ema_val = ema_50_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: 4h RSI < 30 (oversold) + uptrend + volume spike
            if rsi_val < 30 and price > ema_val and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short conditions: 4h RSI > 70 (overbought) + downtrend + volume spike
            elif rsi_val > 70 and price < ema_val and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit conditions: RSI returns to neutral zone (40-60) or trend breaks
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when RSI > 40 or price breaks below EMA
                if rsi_val > 40 or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when RSI < 60 or price breaks above EMA
                if rsi_val < 60 or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_RSI40_4hRSI_1dEMA50_Volume"
timeframe = "1h"
leverage = 1.0