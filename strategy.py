#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator (Jaws/Teeth/Lips) with 1w EMA(34) trend filter and 1d volume spike confirmation.
- Primary timeframe: 12h for signal generation.
- HTF: 1w EMA(34) for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Williams Alligator: Jaws=SMA(13,8), Teeth=SMA(8,5), Lips=SMA(5,3) on 12h median price.
- Entry: Long when Lips > Teeth > Jaws (bullish alignment) AND 1w EMA34 trend bullish AND 1d volume > 1.5 * 20-period volume MA.
         Short when Lips < Teeth < Jaws (bearish alignment) AND 1w EMA34 trend bearish AND 1d volume spike.
- Exit: Opposite Alligator alignment or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator on 12h
    # Median price = (high + low + close) / 3
    median_price = (high + low + close) / 3.0
    
    # Jaws: SMA(13, 8 periods)
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: SMA(8, 5 periods)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: SMA(5, 3 periods)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Get 1w data for EMA(34) trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1w close
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period volume MA on 1d
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 12h volume > 1.5 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 34, 20)  # Need enough bars for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaws_val = jaws[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema_34_val = ema_34_1w_aligned[i]
        curr_close = close[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if vol_spike:
                # Bullish alignment: Lips > Teeth > Jaws
                if lips_val > teeth_val > jaws_val and ema_34_val > 0 and curr_close > ema_34_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish alignment: Lips < Teeth < Jaws
                elif lips_val < teeth_val < jaws_val and ema_34_val > 0 and curr_close < ema_34_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: bearish alignment OR loss of volume confirmation
            if lips_val < teeth_val < jaws_val or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish alignment OR loss of volume confirmation
            if lips_val > teeth_val > jaws_val or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1wEMA34Trend_1dVolumeSpike_v1"
timeframe = "12h"
leverage = 1.0