#!/usr/bin/env python3
"""
1h_RSI_Extreme_4hTrend_VolumeSpike
Hypothesis: On 1h timeframe, RSI extremes (<25 for long, >75 for short) with 4h EMA50 trend filter and volume spike (>2x 24-bar avg) capture mean reversion in range markets while respecting higher timeframe trend. Uses 4h trend to avoid counter-trend trades and volume spike to confirm institutional interest. Targets 15-30 trades/year via tight RSI thresholds and volume confirmation to minimize fee drag. Works in bull markets via trend-aligned mean reversion and in bear markets via oversold bounces in downtrends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 4h for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate RSI(14) on 1h
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume average (24-period = 24h on 1h) for volume spike confirmation
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(24, 50, 14)  # volume MA, 4h EMA, RSI
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(rsi_values[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Get aligned values
        ema_50_4h_val = ema_50_4h_aligned[i]
        rsi_val = rsi_values[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        
        # Volume spike: current volume > 2x 24-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: RSI oversold (<25) with uptrend (close > EMA50) and volume spike
            long_signal = (rsi_val < 25) and (close_val > ema_50_4h_val) and volume_spike
            # Short: RSI overbought (>75) with downtrend (close < EMA50) and volume spike
            short_signal = (rsi_val > 75) and (close_val < ema_50_4h_val) and volume_spike
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit conditions:
            # 1. RSI returns to neutral (>50) - take profit on mean reversion
            if rsi_val > 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: close crosses below EMA50
            elif close_val < ema_50_4h_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit conditions:
            # 1. RSI returns to neutral (<50) - take profit on mean reversion
            if rsi_val < 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: close crosses above EMA50
            elif close_val > ema_50_4h_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "1h_RSI_Extreme_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0