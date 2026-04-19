#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h RSI divergence with 12h volume confirmation and 4h trend filter
# - RSI(14) for overbought/oversold: long when RSI < 30, short when RSI > 70
# - 12h volume > 1.5x 20-period average for conviction
# - 4h EMA(50) trend filter: only take longs when price > EMA50, shorts when price < EMA50
# - Exit on opposite RSI extreme or trend reversal
# - Designed to work in both bull and bear markets by following trend filter
# - Target: 25-40 trades/year to avoid excessive fee drag

name = "4h_RSI_12hVolume_EMA50_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    
    # 12h volume average (20-period)
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # RSI(14) calculation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # 4h EMA(50) for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50[i]) or np.isnan(vol_ma_12h_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 1.5x 12h average volume (scaled)
        # Scale 12h average to 4h: 12h has 3x 4h bars, so divide by 3
        volume_factor = vol_ma_12h_aligned[i] / 3.0
        volume_filter = vol_ma_12h_aligned[i] > 0 and volume[i] > 1.5 * volume_factor
        
        if position == 0:
            # Look for long entry: uptrend (price > EMA50) + oversold RSI + volume
            if close[i] > ema_50[i] and rsi[i] < 30 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend (price < EMA50) + overbought RSI + volume
            elif close[i] < ema_50[i] and rsi[i] > 70 and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on overbought RSI or trend reversal
            if rsi[i] > 70 or close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on oversold RSI or trend reversal
            if rsi[i] < 30 or close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals