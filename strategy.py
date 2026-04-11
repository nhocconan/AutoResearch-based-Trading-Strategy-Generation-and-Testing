#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla Pivot Breakout + Volume Spike + 4h Trend Filter
# - Uses 4h EMA200 for primary trend direction (HTF)
# - 1h Camarilla pivot levels (H3/L3) for breakout entries
# - Volume > 2.0x 20-period average for confirmation
# - Session filter: 08-20 UTC to avoid low-liquidity hours
# - Discrete position sizing: ±0.20 to limit drawdown and fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) to stay within fee drag limits
# - Works in bull markets (breakouts with trend) and bear markets (breakouts against trend filtered by 4h EMA)

name = "1h_4h_camarilla_breakout_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return signals
    
    # Pre-compute 4h EMA200 for trend filter
    close_4h = df_4h['close'].values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # Pre-compute 1h Camarilla pivot levels (based on previous day)
    # Need daily OHLC for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Calculate daily pivots: P = (H+L+C)/3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Camarilla levels: H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    camarilla_h3_1d = close_1d + (high_1d - low_1d) * 1.1 / 4.0
    camarilla_l3_1d = close_1d - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align daily Camarilla levels to 1h timeframe (previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    # Pre-compute 1h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema200_4h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Current price data
        close_current = close[i]
        high_current = high[i]
        low_current = low[i]
        volume_current = volume[i]
        
        # Trend filter: 4h EMA200 direction
        uptrend = close_current > ema200_4h_aligned[i]
        downtrend = close_current < ema200_4h_aligned[i]
        
        # Breakout conditions
        breakout_long = high_current > camarilla_h3_aligned[i]
        breakout_short = low_current < camarilla_l3_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: breakout above H3 + uptrend + volume confirmation
        if breakout_long and uptrend and vol_confirm:
            enter_long = True
        
        # Short: breakout below L3 + downtrend + volume confirmation
        if breakout_short and downtrend and vol_confirm:
            enter_short = True
        
        # Exit conditions: reverse breakout or loss of trend
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below L3 OR trend turns down
            exit_long = (low_current < camarilla_l3_aligned[i]) or (not uptrend)
        elif position == -1:
            # Exit short if price breaks above H3 OR trend turns up
            exit_short = (high_current > camarilla_h3_aligned[i]) or (not downtrend)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.20
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals