#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band squeeze breakout with 1d EMA34 trend filter and volume confirmation.
# Uses Bollinger Band width < 20th percentile to identify low volatility squeeze conditions.
# Breakout occurs when price closes outside Bollinger Bands (20,2) with volume > 1.5x 20-period average.
# Trend filter: price must be above/below 1d EMA34 to align with higher timeframe direction.
# Designed for low trade frequency (~15-25 trades/year) to minimize fee drag and avoid overtrading.
# Bollinger squeeze breakouts work in both trending and ranging markets by capturing volatility expansion.

name = "12h_BollingerSqueeze_Breakout_1dEMA34_VolumeConfirm_v1"
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
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Bollinger Bands (20,2) on 12h timeframe
    if n >= 20:
        close_series = pd.Series(close)
        bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
        bb_std = close_series.rolling(window=20, min_periods=20).std().values
        bb_upper = bb_middle + 2.0 * bb_std
        bb_lower = bb_middle - 2.0 * bb_std
        bb_width = bb_upper - bb_lower
        
        # Calculate Bollinger Band width percentile (20-period lookback for regime)
        if n >= 40:
            bb_width_series = pd.Series(bb_width)
            bb_width_percentile = bb_width_series.rolling(window=40, min_periods=40).rank(pct=True).values * 100
        else:
            bb_width_percentile = np.full(n, np.nan)
    else:
        bb_middle = np.full(n, np.nan)
        bb_upper = np.full(n, np.nan)
        bb_lower = np.full(n, np.nan)
        bb_width = np.full(n, np.nan)
        bb_width_percentile = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if n >= 20:
        vol_ma_20 = np.full(n, np.nan)
        for i in range(20, n):
            vol_ma_20[i] = np.mean(volume[i-20:i])
        volume_confirm = volume > (1.5 * vol_ma_20)
    else:
        volume_confirm = np.full(n, False)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 40  # warmup for BB width percentile
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(bb_width_percentile[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_bb_upper = bb_upper[i]
        curr_bb_lower = bb_lower[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Bollinger squeeze condition: BB width < 20th percentile (low volatility)
        squeeze_condition = bb_width_percentile[i] < 20.0
        
        if position == 0:  # Flat - look for new entries
            # Long: price closes above BB upper, in squeeze, above 1d EMA34, volume confirmation
            if (squeeze_condition and
                curr_close > curr_bb_upper and 
                curr_close > curr_ema_34_1d and 
                vol_conf):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price closes below BB lower, in squeeze, below 1d EMA34, volume confirmation
            elif (squeeze_condition and
                  curr_close < curr_bb_lower and 
                  curr_close < curr_ema_34_1d and 
                  vol_conf):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit: price closes below BB middle (mean reversion) or opposite BB touch
            if curr_close < curr_bb_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above BB middle (mean reversion) or opposite BB touch
            if curr_close > curr_bb_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals