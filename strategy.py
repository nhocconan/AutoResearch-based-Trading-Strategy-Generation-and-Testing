#!/usr/bin/env python3
"""
Hypothesis: 1h Bollinger Band squeeze breakout with 4h trend filter and volume spike.
- Primary timeframe: 1h for precise entry timing on breakouts.
- HTF: 4h EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Bollinger Bands: 20-period, 2.0 std dev on 1h. Squeeze = BB width < 20th percentile of last 50 bars.
- Volume: Current 1h volume > 1.5 * 20-period volume MA to confirm breakout strength.
- Entry: Long when price breaks above upper BB AND 4h EMA50 bullish AND BB squeeze AND volume spike.
         Short when price breaks below lower BB AND 4h EMA50 bearish AND BB squeeze AND volume spike.
- Exit: Opposite band touch (long exits at lower BB, short exits at upper BB) or loss of trend/volume.
- Signal size: 0.20 discrete to limit drawdown and reduce fee churn.
- Session filter: Only trade between 08:00-20:00 UTC to avoid low-liquidity periods.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
This strategy captures explosive moves after low-volatility consolidation, filtered by 4h trend to avoid
counter-trend breakouts, with volume confirmation ensuring institutional participation. Works in both bull
and bear markets by only taking breakout trades in the direction of the 4h trend.
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
    
    # Calculate 1h Bollinger Bands (20, 2.0)
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_stddev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * bb_stddev)
    lower_band = sma - (bb_std * bb_stddev)
    bb_width = upper_band - lower_band
    
    # Bollinger Band squeeze: width < 20th percentile of last 50 bars
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).quantile(0.20).values
    bb_squeeze = bb_width < bb_width_percentile
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    df_4h_close = df_4h['close'].values
    ema_4h = pd.Series(df_4h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period 4h volume MA
    df_4h_volume = df_4h['volume'].values
    vol_ma_4h = pd.Series(df_4h_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 1h
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Volume confirmation: current 1h volume > 1.5 * 20-period 4h volume MA (aligned)
    volume_spike = volume > (1.5 * vol_ma_4h_aligned)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 20)  # Need enough bars for EMA50, BB, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(bb_squeeze[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        bb_squeezed = bb_squeeze[i]
        vol_spike = volume_spike[i]
        ema_val = ema_4h_aligned[i]
        upper_bb = upper_band[i]
        lower_bb = lower_band[i]
        
        if position == 0:
            # Check for entry signals with BB squeeze, volume spike, and session filter
            if bb_squeezed and vol_spike:
                # Bullish breakout: price breaks above upper BB AND 4h EMA50 bullish (close > EMA)
                if curr_close > upper_bb and curr_close > ema_val:
                    signals[i] = 0.20
                    position = 1
                # Bearish breakout: price breaks below lower BB AND 4h EMA50 bearish (close < EMA)
                elif curr_close < lower_bb and curr_close < ema_val:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price touches lower BB OR loss of BB squeeze OR loss of volume spike OR outside session
            if curr_low <= lower_bb or not bb_squeezed or not vol_spike or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price touches upper BB OR loss of BB squeeze OR loss of volume spike OR outside session
            if curr_high >= upper_bb or not bb_squeezed or not vol_spike or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_BB_Squeeze_Breakout_4hEMA50_Trend_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0