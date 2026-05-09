# Hypothesis: Daily volatility breakout using ATR with weekly trend alignment and volume confirmation
# Works in bull/bear markets by capturing volatility expansions while respecting higher timeframe trends
# Target: 20-40 trades/year on daily timeframe to minimize fee drag
# Uses proven volatility breakout concept with institutional-grade volume confirmation

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_VolatilityBreakout_ATR_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate ATR for volatility breakout (using daily data)
    atr_period = 14
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    atr = np.full(n, np.nan)
    for i in range(atr_period, n):
        if not np.isnan(tr[i-atr_period+1:i+1]).any():
            atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # Calculate weekly EMA for trend filter
    ema_weekly = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_1w, ema_weekly)
    
    # Volume confirmation: current volume vs 20-day average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(atr_period, 20)  # Wait for ATR and volume MA to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr[i]) or np.isnan(ema_weekly_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.8 * vol_ma20[i]  # Volume spike confirmation
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        # Calculate breakout levels
        upper_break = close[i-1] + 1.5 * atr[i-1]
        lower_break = close[i-1] - 1.5 * atr[i-1]
        
        if position == 0:
            # Long: upward volatility breakout + above weekly EMA (uptrend) + volume spike
            if (close[i] > upper_break and 
                close[i] > ema_weekly_aligned[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: downward volatility breakout + below weekly EMA (downtrend) + volume spike
            elif (close[i] < lower_break and 
                  close[i] < ema_weekly_aligned[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes below previous close or volatility contraction
            if close[i] < close[i-1] or volume[i] < 0.6 * vol_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above previous close or volatility contraction
            if close[i] > close[i-1] or volume[i] < 0.6 * vol_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals