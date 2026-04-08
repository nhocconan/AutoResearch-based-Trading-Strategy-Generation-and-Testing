#!/usr/bin/env python3
"""
12h_1d_cci_volatility_breakout_v1
Hypothesis: On 12h timeframe, price breaking above/below 20-period CCI bands with volatility expansion and daily trend alignment captures sustained moves. Daily trend filter avoids counter-trend breakouts in ranging markets. Volatility filter (ATR ratio) ensures breaks occur during expanded volatility regimes. Designed for trending markets while avoiding false breakouts in low volatility environments.
- Long: CCI(20) > +100 + ATR(14)/ATR(50) > 1.2 + daily uptrend (price > EMA50)
- Short: CCI(20) < -100 + ATR(14)/ATR(50) > 1.2 + daily downtrend (price < EMA50)
- Exit: CCI crosses back through zero or daily trend reversal
- Position sizing: 0.25 long, -0.25 short
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_cci_volatility_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Daily EMA(50) for trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema_50_1d
    trend_1d_down = close_1d < ema_50_1d
    
    # Forward fill trend
    trend_1d_up_series = pd.Series(trend_1d_up)
    trend_1d_down_series = pd.Series(trend_1d_down)
    trend_1d_up_ffilled = trend_1d_up_series.ffill().values
    trend_1d_down_ffilled = trend_1d_down_series.ffill().values
    
    # Align 1d trend to 12h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up_ffilled)
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down_ffilled)
    
    # Calculate CCI(20) on 12h data
    typical_price = (high + low + close) / 3.0
    tp_ma = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    tp_mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    # Avoid division by zero
    tp_mad = np.where(tp_mad == 0, 1e-10, tp_mad)
    cci = (typical_price - tp_ma) / (0.015 * tp_mad)
    
    # Calculate ATR ratio for volatility filter
    # ATR(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # ATR(50)
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    # Avoid division by zero
    atr_ratio = np.where(atr_50 == 0, 0, atr_14 / atr_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(cci[i]) or np.isnan(atr_ratio[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI crosses below zero OR daily trend turns down
            if (cci[i] < 0) or trend_1d_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: CCI crosses above zero OR daily trend turns up
            if (cci[i] > 0) or trend_1d_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: CCI > +100 + volatility expansion + daily uptrend
            if (cci[i] > 100) and (atr_ratio[i] > 1.2) and trend_1d_up_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: CCI < -100 + volatility expansion + daily downtrend
            elif (cci[i] < -100) and (atr_ratio[i] > 1.2) and trend_1d_down_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals