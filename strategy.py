#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Mean Reversion with 1w EMA50 trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In bull regime (price > 1w EMA50),
# we go long when %R crosses above -80 from below. In bear regime (price < 1w EMA50),
# we go short when %R crosses below -20 from above. Volume spike confirms momentum.
# This combines mean reversion entries with higher timeframe trend filtering for robustness
# in both bull and bear markets.

name = "6h_WilliamsR_1wTrend_VolumeSpike_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate Williams %R (14-period) on primary timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate volume regime: current 6h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        wr = williams_r[i]
        ema_trend = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        close_val = close[i]
        
        # Skip if any value is NaN
        if np.isnan(wr) or np.isnan(ema_trend) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: bull if close > 1w EMA50, bear if close < 1w EMA50
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Williams %R cross conditions with momentum
        wr_prev = williams_r[i-1] if i > 0 else -50
        
        # Long: %R crosses above -80 from below (oversold recovery) with volume spike in bull regime
        long_entry = is_bull_regime and (wr_prev < -80) and (wr >= -80) and vol_spike
        
        # Short: %R crosses below -20 from above (overbought rejection) with volume spike in bear regime
        short_entry = is_bear_regime and (wr_prev > -20) and (wr <= -20) and vol_spike
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on %R crossing above -20 (overbought) or regime change to bear
            if wr >= -20 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on %R crossing below -80 (oversold) or regime change to bull
            if wr <= -80 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals