#!/usr/bin/env python3
"""
6h_RSI2_MeanReversion_1dTrendFilter_VolumeSpike
Hypothesis: On 6h timeframe, use 2-period RSI for extreme mean reversion signals (RSI<10 for long, RSI>90 for short) only when aligned with 1d EMA50 trend and confirmed by volume spike (>2.0x 20-period average). This strategy targets short-term reversals within the prevailing daily trend, exploiting overextended moves that tend to revert. Discrete sizing 0.25. Target ~15-25 trades/year to minimize fee drag while capturing high-conviction mean reversion opportunities in both bull and bear markets.
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
    open_time = prices['open_time'].values
    
    # Session filter: UTC 8-20 for institutional activity
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # RSI(2) on 6h for mean reversion signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of RSI(2) (2), 1d EMA (50), volume MA (20)
    start_idx = max(2, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(rsi[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        rsi_val = rsi[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average (strict for signal quality)
        volume_confirmed = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: RSI<10 (extremely oversold) + uptrend (close > EMA50_1d) + volume confirmation
            long_signal = (rsi_val < 10) and (close_val > ema_50_1d_val) and volume_confirmed
            # Short: RSI>90 (extremely overbought) + downtrend (close < EMA50_1d) + volume confirmation
            short_signal = (rsi_val > 90) and (close_val < ema_50_1d_val) and volume_confirmed
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: RSI crosses above 50 (mean reversion complete) or trend reversal
            if rsi_val > 50 or close_val < ema_50_1d_val:
                signals[i] = 0.0
                position = 0
            # Exit: close below prior low (failed mean reversion)
            elif i >= 2 and close_val < low[i-1]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: RSI crosses below 50 (mean reversion complete) or trend reversal
            if rsi_val < 50 or close_val > ema_50_1d_val:
                signals[i] = 0.0
                position = 0
            # Exit: close above prior high (failed mean reversion)
            elif i >= 2 and close_val > high[i-1]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_RSI2_MeanReversion_1dTrendFilter_VolumeSpike"
timeframe = "6h"
leverage = 1.0