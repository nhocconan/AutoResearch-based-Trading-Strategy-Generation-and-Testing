#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d ADX trend filter and volume confirmation.
# Long when BB width < 20th percentile (squeeze) AND price breaks above upper band with 1d ADX > 25 and volume > 2x 24-bar average.
# Short when BB width < 20th percentile (squeeze) AND price breaks below lower band with 1d ADX > 25 and volume confirmation.
# Uses discrete sizing 0.25. ATR-based stoploss (signal→0 when price moves against position by 2.5*ATR).
# Primary timeframe: 6h, HTF: 1d for ADX trend filter.
# Target: 50-150 total trades over 4 years (12-38/year) to minimize fee drag.
# Session filter: 08-20 UTC to reduce noise trades.

name = "6h_BB_Squeeze_ADX_Trend_VolumeBreakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for 08-20 UTC filter
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend filter
    # TR calculation
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['low'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    # +DM and -DM
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low'].shift()) - pd.Series(df_1d['low'])
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    # Smooth TR, +DM, -DM
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_1d = 100 * (pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d)
    minus_di_1d = 100 * (pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate ATR(14) for 6h data (for stoploss and BB)
    tr1_6h = high[1:] - low[1:]
    tr2_6h = np.abs(high[1:] - close[:-1])
    tr3_6h = np.abs(low[1:] - close[:-1])
    tr_6h = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))])
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Bollinger Bands (20, 2) on 6h close
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Calculate 20th percentile of BB width for squeeze condition (using expanding window)
    bb_width_percentile = np.full_like(bb_width, np.nan)
    for i in range(20, len(bb_width)):
        bb_width_percentile[i] = np.percentile(bb_width[20:i+1], 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    start_idx = 50  # warmup for ADX, ATR, BB, and volume
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(atr_6h[i]) or 
            np.isnan(bb_width[i]) or np.isnan(bb_width_percentile[i]) or
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0x 24-bar average (tight to reduce trades)
        vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values[i]
        if vol_ma <= 0:
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma * 2.0)
        
        # Bollinger Band squeeze condition: width < 20th percentile of historical width
        squeeze_condition = bb_width[i] < bb_width_percentile[i]
        
        # Breakout conditions
        breakout_up = curr_close > bb_upper[i-1]  # break above upper band (using previous bar)
        breakout_down = curr_close < bb_lower[i-1]  # break below lower band (using previous bar)
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_1d_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: BB squeeze AND breakout up AND trending AND volume confirmation
            if (squeeze_condition and 
                breakout_up and 
                trending and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: BB squeeze AND breakout down AND trending AND volume confirmation
            elif (squeeze_condition and 
                  breakout_down and 
                  trending and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close < entry_price - 2.5 * atr_6h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters BB (breakout failed) OR volatility expands (squeeze ends)
            elif (curr_close < bb_middle[i] or 
                  bb_width[i] > bb_width_percentile[i] * 1.5):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr_6h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters BB (breakout failed) OR volatility expands (squeeze ends)
            elif (curr_close > bb_middle[i] or 
                  bb_width[i] > bb_width_percentile[i] * 1.5):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals