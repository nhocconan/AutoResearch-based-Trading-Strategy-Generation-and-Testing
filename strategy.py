#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d Trend Filter and Volume Confirmation
# Strategy exploits low volatility contractions (Bollinger Band squeeze) followed by breakouts.
# Long when: BB width < 20th percentile (squeeze), price breaks above upper BB, 1d EMA50 uptrend, volume > 2x average.
# Short when: BB width < 20th percentile (squeeze), price breaks below lower BB, 1d EMA50 downtrend, volume > 2x average.
# Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-35 trades/year (~50-140 total over 4 years).
# Works in bull markets via upside breakouts and bear markets via downside breakouts after volatility contraction.

name = "6h_BollingerSqueeze_Breakout_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = close_series.rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_middle + (bb_std * bb_std_dev)
    bb_lower = bb_middle - (bb_std * bb_std_dev)
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band Width percentile (20-period lookback for squeeze detection)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Volume confirmation: >2.0x 20-bar average volume (strict filter to reduce trades)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(bb_period, 50, 20)  # BB(20), 1d EMA(50), volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_middle[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(bb_width_percentile[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        squeeze_condition = bb_width_percentile[i] < 20.0  # BB width in lowest 20% = squeeze
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: BB squeeze + price breaks above upper BB + 1d uptrend + volume spike
            if (squeeze_condition and 
                price > bb_upper[i] and 
                close[i-1] <= bb_upper[i-1] and  # Confirm breakout (was below or at upper bar)
                ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and  # 1d EMA50 rising
                vol_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: BB squeeze + price breaks below lower BB + 1d downtrend + volume spike
            elif (squeeze_condition and 
                  price < bb_lower[i] and 
                  close[i-1] >= bb_lower[i-1] and  # Confirm breakout (was above or at lower bar)
                  ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and  # 1d EMA50 falling
                  vol_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or mean reversion to middle BB
            # ATR-based stoploss: 2.5 * ATR below entry (using 6h ATR)
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price - 2.5 * atr_val
            # Exit if stoploss hit or price reverts to middle BB (mean reversion)
            if price < stop_loss or price < bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on stoploss or mean reversion to middle BB
            # ATR-based stoploss: 2.5 * ATR above entry
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price + 2.5 * atr_val
            # Exit if stoploss hit or price reverts to middle BB (mean reversion)
            if price > stop_loss or price > bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals