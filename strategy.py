#!/usr/bin/env python3
"""
Hypothesis: 1d Williams %R(14) with 1w EMA(34) trend filter and volume spike confirmation.
- Primary timeframe: 1d for entries/exits.
- HTF: 1w EMA(34) for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Volume: Current 1d volume > 2.0 * 20-period volume MA to confirm breakout strength.
- Entry: Long when Williams %R crosses above -80 from below AND 1w EMA34 bullish AND volume spike.
         Short when Williams %R crosses below -20 from above AND 1w EMA34 bearish AND volume spike.
- Exit: Opposite Williams %R crossover or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
- Works in bull/bear: Williams %R identifies oversold/overbought conditions; EMA filter ensures trend alignment.
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
    
    # Calculate Williams %R(14) on 1d
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    williams_r = williams_r.values
    
    # Get 1w data for EMA(34) trend and volume MA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1w close
    ema_34 = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 1w
    vol_ma_1w = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 1d
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Volume confirmation: current 1d volume > 2.0 * 20-period 1w volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1w_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 14)  # Need enough 1w bars for EMA34 and enough 1d bars for Williams %R
    
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
        
        # Williams %R crossover signals
        williams_r_cross_up = (williams_r[i] > -80) and (williams_r[i-1] <= -80)
        williams_r_cross_down = (williams_r[i] < -20) and (williams_r[i-1] >= -20)
        
        if position == 0:
            # Check for entry signals with volume spike
            if curr_volume_spike:
                # Bullish entry: Williams %R crosses above -80 from below AND 1w EMA34 bullish (close > EMA34)
                if williams_r_cross_up and curr_close > ema_34_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Williams %R crosses below -20 from above AND 1w EMA34 bearish (close < EMA34)
                elif williams_r_cross_down and curr_close < ema_34_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -80 OR loss of volume confirmation
            if williams_r[i] < -80 or not curr_volume_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -20 OR loss of volume confirmation
            if williams_r[i] > -20 or not curr_volume_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR14_1wEMA34Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0