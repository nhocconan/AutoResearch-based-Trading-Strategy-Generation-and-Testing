#!/usr/bin/env python3
# Hypothesis: 6h Williams %R with 1-week EMA50 trend filter and 1-day volume spike confirmation.
# Long when Williams %R(14) < -80 (oversold) AND price > 1w EMA50 (uptrend) AND volume > 2.0x 1d average
# Short when Williams %R(14) > -20 (overbought) AND price < 1w EMA50 (downtrend) AND volume > 2.0x 1d average
# Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts) OR trend reversal.
# Uses 6h timeframe (target: 50-150 total trades over 4 years = 12-37/year) with 1w trend filter for BTC/ETH resilience in bull/bear markets.
# Williams %R captures mean reversion extremes; 1w EMA50 filters primary trend; 1d volume spike confirms exhaustion.

name = "6h_WilliamsR_1wEMA50_1dVolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Williams %R calculation (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Williams %R(14) on 6h data (using previous bar's data to avoid look-ahead)
    if len(high_6h) >= 14:
        highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().shift(1).values
        lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().shift(1).values
        williams_r = -100 * (highest_high - close_6h) / (highest_high - lowest_low)
        # Handle division by zero when highest_high == lowest_low
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    else:
        williams_r = np.full_like(high_6h, np.nan)
    
    # Align Williams %R to 6h timeframe (already aligned since calculated on 6h)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Get 1w data for EMA50 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate volume average on 1d (20-period) for spike confirmation
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume filter: current 6h volume > 2.0x 1d average volume (scaled to 6h)
    # Approximate: 1d volume ≈ 4 * 6h volume (since 24h/6h = 4)
    volume_filter = volume > (0.5 * vol_ma_1d_aligned)  # 2.0x/4 = 0.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for EMA and volume
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND price > 1w EMA50 AND volume spike
            if williams_r_aligned[i] < -80 and close[i] > ema50_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) AND price < 1w EMA50 AND volume spike
            elif williams_r_aligned[i] > -20 and close[i] < ema50_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R > -50 (reverting from oversold) OR trend reversal (close < 1w EMA50)
            if williams_r_aligned[i] > -50 or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R < -50 (reverting from overbought) OR trend reversal (close > 1w EMA50)
            if williams_r_aligned[i] < -50 or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals