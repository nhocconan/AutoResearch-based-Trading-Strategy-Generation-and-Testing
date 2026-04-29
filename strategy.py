#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation
# In low volatility regimes (BB width < 20th percentile), price often breaks out strongly
# We trade breakouts in the direction of the 1d EMA50 trend with volume confirmation (>1.5x average)
# This captures explosive moves after consolidation phases, works in both bull and bear markets
# Target: 12-37 trades/year on 6h timeframe to minimize fee drag while capturing high-momentum breakouts

name = "6h_BBandSqueeze_Breakout_1dEMA50_VolumeConfirm_v1"
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
    
    # Get 1d data for EMA50 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Bollinger Bands (20, 2.0) on 6h data
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Calculate BB width percentile (20-period lookback) for squeeze detection
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=20, min_periods=20).rank(pct=True).values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = 20  # warmup for BB and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bb_middle[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(bb_width_percentile[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_bb_width_pct = bb_width_percentile[i]
        curr_vol_ma = vol_ma_20[i]
        curr_atr = atr[i]
        curr_bb_upper = bb_upper[i]
        curr_bb_lower = bb_lower[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or price closes below middle band (mean reversion)
            if curr_close < entry_price - 1.5 * curr_atr or curr_close < bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or price closes above middle band (mean reversion)
            if curr_close > entry_price + 1.5 * curr_atr or curr_close > bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Squeeze condition: BB width in lower 20th percentile (low volatility)
            squeeze_condition = curr_bb_width_pct <= 0.20
            
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirm = curr_volume > 1.5 * curr_vol_ma
            
            # Long entry: price breaks above upper BB AND in uptrend (price > 1d EMA50) AND squeeze + volume
            if curr_close > curr_bb_upper and curr_close > curr_ema50_1d and squeeze_condition and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
            # Short entry: price breaks below lower BB AND in downtrend (price < 1d EMA50) AND squeeze + volume
            elif curr_close < curr_bb_lower and curr_close < curr_ema50_1d and squeeze_condition and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals