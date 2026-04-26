#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_FundingExtreme_v2
Hypothesis: 12h Camarilla pivot R1/S1 breakout with 1-day EMA34 trend filter and extreme funding rate contrarian signal.
Only trade breakouts in direction of 1-day trend when funding rate is extreme (>0.1% or <-0.1%).
Funding extremes indicate overextended leveraged positions likely to reverse. Uses discrete 0.25 position size.
Target: 12-37 trades/year (50-150 over 4 years) by requiring confluence of breakout, trend, and extreme funding.
Works in bull/bear via 1-day trend filter: only takes longs in uptrend, shorts in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for HTF trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    htf_trend = np.where(close > ema_34_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate Camarilla pivot levels from 1d data
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    R1_1d = typical_price_1d + (1.1/12) * (df_1d['high'] - df_1d['low'])  # R1 level
    S1_1d = typical_price_1d - (1.1/12) * (df_1d['high'] - df_1d['low'])  # S1 level
    
    # Align Camarilla levels to 12h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d.values)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d.values)
    
    # Load funding rate data (8h) and align to 12h
    try:
        df_8h = get_htf_data(prices, '8h')
        # Funding rate is typically in the data as 'funding_rate' column
        if 'funding_rate' in df_8h.columns:
            funding_rate = df_8h['funding_rate'].values
        else:
            # Fallback: use zero if funding rate not available (should not happen on Binance)
            funding_rate = np.zeros(len(df_8h))
        funding_rate_aligned = align_htf_to_ltf(prices, df_8h, funding_rate)
    except:
        # If funding rate data not available, disable filter (neutral)
        funding_rate_aligned = np.zeros(n)
    
    # Volume confirmation: volume > 2.0x 20-period average (much tighter)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1d EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or
            np.isnan(funding_rate_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (much tighter: 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Funding rate EXTREME condition (contrarian signal) - only trade on extremes
        funding_extreme_long = funding_rate_aligned[i] < -0.001  # < -0.1%
        funding_extreme_short = funding_rate_aligned[i] > 0.001   # > +0.1%
        funding_extreme = funding_extreme_long or funding_extreme_short
        
        # Breakout conditions with trend filter and EXTREME funding filter
        if htf_trend[i] == 1:  # Uptrend on 1d
            # Long breakout above R1 with volume spike, EXTREME funding long bias
            if close[i] > R1_1d_aligned[i] and volume_spike and funding_extreme_long:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Exit long if price falls below S1 (reversal signal) or funding turns extremely short
            elif position == 1 and (close[i] < S1_1d_aligned[i] or funding_extreme_short):
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        elif htf_trend[i] == -1:  # Downtrend on 1d
            # Short breakdown below S1 with volume spike, EXTREME funding short bias
            if close[i] < S1_1d_aligned[i] and volume_spike and funding_extreme_short:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit short if price rises above R1 (reversal signal) or funding turns extremely long
            elif position == -1 and (close[i] > R1_1d_aligned[i] or funding_extreme_long):
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Should not happen with our trend calculation
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_FundingExtreme_v2"
timeframe = "12h"
leverage = 1.0