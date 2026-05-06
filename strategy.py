#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily volatility contraction/expansion with volume confirmation
# Low volatility (ATR contraction) followed by expansion with volume indicates breakout
# Uses daily ATR contraction filter (ATR < 0.8 * 20-day average) to identify low vol periods
# Breakout entry when price breaks above/below 12h high/low of prior 3 bars with volume > 1.5x 20-period average
# Works in bull/bear markets: volatility expansion captures breakouts in both directions
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_1dATRContraction_VolumeBreakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily ATR for volatility filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # True Range for daily ATR
    tr1 = df_1d['high'].values[1:] - df_1d['low'].values[1:]
    tr2 = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
    tr3 = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
    tr_daily = np.concatenate([[np.nan], np.maximum(np.maximum(tr1, tr2), tr3)])
    atr_14_daily = pd.Series(tr_daily).rolling(window=14, min_periods=14).mean().values
    atr_ma_20_daily = pd.Series(atr_14_daily).rolling(window=20, min_periods=20).mean().values
    
    # Volatility filter: daily ATR < 0.8 * 20-day average ATR (contraction)
    vol_filter_daily = atr_14_daily < (0.8 * atr_ma_20_daily)
    vol_filter_daily_aligned = align_htf_to_ltf(prices, df_1d, vol_filter_daily)
    
    # Volume confirmation: >1.5x 20-period average (moderate threshold to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(vol_filter_daily_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Lookback period for 12h high/low (prior 3 bars)
            lookback_start = max(0, i - 3)
            lookback_end = i
            if lookback_end > lookback_start:
                period_high = np.max(high[lookback_start:lookback_end])
                period_low = np.min(low[lookback_start:lookback_end])
                
                # Long breakout: price breaks above period high with volume and vol contraction
                if close[i] > period_high and volume_filter[i] and vol_filter_daily_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakout: price breaks below period low with volume and vol contraction
                elif close[i] < period_low and volume_filter[i] and vol_filter_daily_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price breaks below period low (failed breakout) or time-based exit
            lookback_start = max(0, i - 3)
            lookback_end = i
            if lookback_end > lookback_start:
                period_low = np.min(low[lookback_start:lookback_end])
                if close[i] < period_low:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above period high (failed breakdown) or time-based exit
            lookback_start = max(0, i - 3)
            lookback_end = i
            if lookback_end > lookback_start:
                period_high = np.max(high[lookback_start:lookback_end])
                if close[i] > period_high:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals