#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R extreme reversal with 1d trend filter and volume spike
# Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
# Long when Williams %R < -80 (oversold) AND 1d close > 1d EMA50 (uptrend) AND volume > 2x 20-bar avg
# Short when Williams %R > -20 (overbought) AND 1d close < 1d EMA50 (downtrend) AND volume > 2x 20-bar avg
# Exit when Williams %R returns to -50 (mean reversion) or volume drops
# Target: 20-50 trades/year via extreme readings + volume confirmation + trend filter
# Works in bull markets by buying oversold dips in uptrends, in bear markets by selling overbought rallies in downtrends

name = "4h_WilliamsR_Extreme_1dTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R(14) on 4h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Volume confirmation: >2x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need sufficient history for Williams %R(14) + EMA50 + volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r[i]
        trend_up = close[i] > ema_50_1d_aligned[i]  # 4h close above 1d EMA50
        trend_down = close[i] < ema_50_1d_aligned[i]  # 4h close below 1d EMA50
        vol_spike = volume_spike[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when Williams %R < -80 (oversold) AND uptrend AND volume spike
            if wr < -80 and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short when Williams %R > -20 (overbought) AND downtrend AND volume spike
            elif wr > -20 and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when Williams %R >= -50 (mean reversion) or no volume spike
            if wr >= -50 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when Williams %R <= -50 (mean reversion) or no volume spike
            if wr <= -50 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals