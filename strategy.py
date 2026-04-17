#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R + 1d EMA50 trend filter + volume spike confirmation
- Williams %R(14) identifies overbought/oversold conditions with proven mean reversion edge
- 1d EMA50 provides strong trend filter (bullish when price > EMA50, bearish when price < EMA50)
- Volume spike (2.0x 20-period MA) confirms institutional participation at turning points
- Long: Williams %R < -80 (oversold) + price > 1d EMA50 (uptrend) + volume spike
- Short: Williams %R > -20 (overbought) + price < 1d EMA50 (downtrend) + volume spike
- Exit: Williams %R crosses back above -50 (for longs) or below -50 (for shorts)
- Discrete position sizing (0.25) minimizes fee churn
- Target: 12-30 trades/year per symbol (~50-120 total over 4 years)
- Works in bull markets (buying oversold dips in uptrend) and bear markets (selling overbought rallies in downtrend)
- Williams %R is effective in ranging/choppy markets which appear in both bull and bear regimes
"""

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
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams %R on 6h (primary timeframe)
    def williams_r(high_arr, low_arr, close_arr, window=14):
        highest_high = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        wr = -100 * (highest_high - close_arr) / (highest_high - lowest_low)
        # Handle division by zero when highest_high == lowest_low
        wr = np.where((highest_high - lowest_low) == 0, -50, wr)
        return wr
    
    wr_14 = williams_r(high, low, close, 14)
    
    # Volume average (20-period) on 6h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(wr_14[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        wr = wr_14[i]
        ema_trend = ema50_1d_aligned[i]
        vol_ma = volume_ma[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for mean reversion entries with volume confirmation and trend alignment
            # Long: oversold (WR < -80) + price > 1d EMA50 (uptrend) + volume spike
            if wr < -80 and price > ema_trend and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: overbought (WR > -20) + price < 1d EMA50 (downtrend) + volume spike
            elif wr > -20 and price < ema_trend and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses back above -50 (momentum weakening)
            if wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses back below -50 (momentum weakening)
            if wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0