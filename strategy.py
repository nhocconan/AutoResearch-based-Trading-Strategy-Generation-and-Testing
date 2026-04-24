#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla pivot breakout with 1d trend filter and volume spike confirmation.
- Primary timeframe: 1h for precise entry/exit timing.
- HTF: 1d Camarilla H3/L3 levels for breakout signals (price > H3 = bullish, price < L3 = bearish).
- Trend: 1d EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Volume: Current 1h volume > 2.0 * 24-period 1d volume MA to capture institutional interest.
- Entry: Long when price breaks above H3 AND 1d EMA34 bullish AND volume spike.
         Short when price breaks below L3 AND 1d EMA34 bearish AND volume spike.
- Exit: Price retreats to H4/L4 levels (Camarilla mean reversion) or loss of volume confirmation.
- Signal size: 0.20 discrete to limit drawdown and reduce fee churn.
- Session filter: Only trade between 08:00-20:00 UTC to avoid low-liquidity periods.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
This strategy uses Camarilla pivots as dynamic support/resistance levels, which work well in both
trending and ranging markets. The 1d trend filter ensures we only trade with the higher timeframe
bias, while volume spikes confirm institutional participation in the breakout.
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
    
    # Get 1d data for Camarilla pivots and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 24-period 1d volume MA for volume confirmation
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=24, min_periods=24).mean().values
    
    # Calculate 1d Camarilla pivots (H3, L3, H4, L4)
    # Camarilla formulas: 
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.0 * (high - low)
    # L3 = close - 1.0 * (high - low)
    # L4 = close - 1.5 * (high - low)
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close_vals = df_1d['close'].values
    
    camarilla_h3 = df_1d_close_vals + 1.0 * (df_1d_high - df_1d_low)
    camarilla_l3 = df_1d_close_vals - 1.0 * (df_1d_high - df_1d_low)
    camarilla_h4 = df_1d_close_vals + 1.5 * (df_1d_high - df_1d_low)
    camarilla_l4 = df_1d_close_vals - 1.5 * (df_1d_high - df_1d_low)
    
    # Align HTF indicators to 1h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: current 1h volume > 2.0 * 24-period 1d volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 24)  # Need enough bars for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_val = ema_1d_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike and session filter
            if volume_spike[i]:
                # Bullish: Price breaks above H3 AND 1d EMA34 bullish (close > EMA)
                if curr_close > camarilla_h3_aligned[i] and curr_close > ema_val:
                    signals[i] = 0.20
                    position = 1
                # Bearish: Price breaks below L3 AND 1d EMA34 bearish (close < EMA)
                elif curr_close < camarilla_l3_aligned[i] and curr_close < ema_val:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: Price retreats to H4 (mean reversion) OR loss of volume confirmation OR outside session
            if curr_close < camarilla_h4_aligned[i] or not volume_spike[i] or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Price retreats to L4 (mean reversion) OR loss of volume confirmation OR outside session
            if curr_close > camarilla_l4_aligned[i] or not volume_spike[i] or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0