#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R Extreme + 1d EMA34 Trend Filter + Volume Spike
# Williams %R(14) < -80 = oversold (long signal), > -20 = overbought (short signal)
# Only trade in direction of 1d EMA34 trend: long when close > EMA34, short when close < EMA34
# Volume confirmation: current volume > 2.0x 20-bar average to avoid low-liquidity false signals
# Exit when Williams %R returns to neutral range (-50) or volume drops
# Target: 12-37 trades/year via extreme readings + volume filter reducing false signals
# Williams %R is a momentum oscillator that works well in both bull and bear markets
# when combined with trend filter and volume confirmation to avoid whipsaws

name = "12h_WilliamsR_Extreme_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d close
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R(14) on 12h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_34_val = ema_34_1d_aligned[i]
        wr = williams_r[i]
        close_val = close[i]
        
        # Determine trend direction from 1d EMA34
        uptrend = close_val > ema_34_val
        downtrend = close_val < ema_34_val
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when Williams %R < -80 (oversold) AND uptrend AND volume confirmation
            if wr < -80 and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Williams %R > -20 (overbought) AND downtrend AND volume confirmation
            elif wr > -20 and downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when Williams %R >= -50 or no volume
            if wr >= -50 or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when Williams %R <= -50 or no volume
            if wr <= -50 or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals