#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 12h Trend Filter and Volume Spike
# Bollinger Squeeze (BB Width < 20th percentile) indicates low volatility primed for breakout
# Breakout direction confirmed by 12h EMA50 trend and volume > 2.0x 20-period average
# Only takes longs when price breaks above upper BB in uptrend (price > 12h EMA50)
# Only takes shorts when price breaks below lower BB in downtrend (price < 12h EMA50)
# Designed for ~20-35 trades/year on 6h timeframe to minimize fee drag while capturing explosive moves
# Works in both bull and bear markets via 12h trend filter - only trades in trend direction

name = "6h_BollingerSqueeze_Breakout_12hEMA50_VolumeSpike_v1"
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
    
    # Get 12h data for EMA50 trend filter (HTF = 12h)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Bollinger Bands (20, 2.0) on 6h data
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Calculate Bollinger Band Width percentile (20-period lookback) for squeeze detection
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=20, min_periods=20).rank(pct=True).values
    
    # Calculate 20-period average volume for confirmation (on 6h data)
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
    
    start_idx = 20  # Bollinger Bands and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(bb_width_percentile[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50_12h = ema_50_12h_aligned[i]
        curr_bb_upper = bb_upper[i]
        curr_bb_lower = bb_lower[i]
        curr_bb_width_pct = bb_width_percentile[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        curr_atr = atr[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or price re-enters Bollinger Bands (mean reversion)
            if curr_close < entry_price - 2.0 * curr_atr or curr_close < bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or price re-enters Bollinger Bands (mean reversion)
            if curr_close > entry_price + 2.0 * curr_atr or curr_close > bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Bollinger Squeeze condition: BB Width < 20th percentile (low volatility)
            squeeze_condition = curr_bb_width_pct < 0.20
            
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Long entry: price breaks above upper BB in uptrend (price > 12h EMA50)
            if squeeze_condition and vol_confirm and curr_close > curr_ema50_12h:
                if curr_close > curr_bb_upper:  # Breakout above upper BB
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
            # Short entry: price breaks below lower BB in downtrend (price < 12h EMA50)
            elif squeeze_condition and vol_confirm and curr_close < curr_ema50_12h:
                if curr_close < curr_bb_lower:  # Breakout below lower BB
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals