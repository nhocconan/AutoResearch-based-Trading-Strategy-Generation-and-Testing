#!/usr/bin/env python3
"""
Hypothesis: 1h RSI(2) extreme reversals with 4h trend filter and volume spike confirmation.
- Primary timeframe: 1h for execution, HTF: 4h for trend direction (EMA50), 1d for volume regime.
- RSI(2) < 10 = oversold, RSI(2) > 90 = overbought on 1h.
- In 4h uptrend (close > EMA50): long RSI(2) < 10 with volume spike.
- In 4h downtrend (close < EMA50): short RSI(2) > 90 with volume spike.
- Volume spike: current volume > 2.0 * 20-period volume MA (1h) to avoid low-volume false signals.
- Discrete signal size: 0.20 to limit drawdown and reduce fee churn.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
- Session filter: 08-20 UTC to avoid low-liquidity Asian session noise.
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
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 4h close
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume regime (optional filter - we'll use 1h volume spike instead)
    # But we can use 1d average volume as baseline if needed
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period volume MA on 1h for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Calculate RSI(2) on 1h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for 4h EMA, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_val = rsi[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        ema_50 = ema_50_4h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Check for entry signals
            if vol_spike:
                if curr_close > ema_50:  # 4h uptrend: look for longs on RSI(2) extreme oversold
                    if rsi_val < 10:
                        signals[i] = 0.20
                        position = 1
                elif curr_close < ema_50:  # 4h downtrend: look for shorts on RSI(2) extreme overbought
                    if rsi_val > 90:
                        signals[i] = -0.20
                        position = -1
        elif position == 1:
            # Long exit: RSI(2) > 50 (mean reversion) OR trend breaks down
            if rsi_val > 50 or curr_close < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI(2) < 50 (mean reversion) OR trend breaks up
            if rsi_val < 50 or curr_close > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI2_Extremes_4hEMA50Trend_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0