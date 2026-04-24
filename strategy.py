#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with 1d EMA(34) trend filter and 1d volume spike confirmation.
- Primary timeframe: 6h for entries/exits.
- HTF: 1d EMA(34) for trend direction (bullish if price > EMA34, bearish if price < EMA34).
- Volume: Current 6h volume > 2.0 * 20-period volume MA from 1d to avoid false signals.
- Entry: Long when Williams %R(14) crosses above -20 (oversold recovery) AND 1d EMA34 bullish AND volume spike.
         Short when Williams %R(14) crosses below -80 (overbought breakdown) AND 1d EMA34 bearish AND volume spike.
- Exit: Opposite Williams %R cross or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Williams %R is effective in ranging markets (common in 2025+ BTC/ETH) and catches reversals in bear rallies.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R(14) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    denom = highest_high - lowest_low
    denom = np.where(denom == 0, 1e-10, denom)
    williams_r = -100 * (highest_high - close) / denom
    
    # Get 1d data for EMA(34) trend and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d close
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 1d
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 6h volume > 2.0 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 14)  # Need enough 1d bars for EMA34 and 6h bars for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_34_val = ema_34_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_spike = volume_spike[i]
        prev_williams_r = williams_r[i-1] if i > 0 else williams_r[i]
        curr_williams_r = williams_r[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if curr_volume_spike:
                # Bullish entry: Williams %R crosses above -20 (recovery from oversold) AND 1d EMA34 bullish
                if prev_williams_r <= -20 and curr_williams_r > -20 and ema_34_val > 0 and curr_close > ema_34_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Williams %R crosses below -80 (breakdown from overbought) AND 1d EMA34 bearish
                elif prev_williams_r >= -80 and curr_williams_r < -80 and ema_34_val > 0 and curr_close < ema_34_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -80 (overbought) OR loss of volume confirmation
            if prev_williams_r >= -80 and curr_williams_r < -80 or not curr_volume_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -20 (oversold) OR loss of volume confirmation
            if prev_williams_r <= -20 and curr_williams_r > -20 or not curr_volume_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dEMA34Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0