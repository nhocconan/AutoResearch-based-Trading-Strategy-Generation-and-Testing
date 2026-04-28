#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d EMA34 trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In ranging markets (chop > 61.8),
# we mean revert from extremes. In trending markets (chop < 38.2), we follow the 1d EMA34 trend.
# Uses 12h timeframe for low trade frequency (~15-25 trades/year) to minimize fee drag.
# Volume confirmation ensures breakouts have participation. Designed for both bull and bear markets
# via adaptive regime filtering. Target: 50-150 total trades over 4 years (12-37/year). Size: 0.25.

name = "12h_WilliamsR_1dEMA34_Chop_Regime_VolumeConfirm_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid TypeError
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA34 (trend filter) and Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Williams %R (14-period)
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d Choppiness Index (14-period) for regime filter
    atr_1d = pd.Series(np.maximum(high_1d - low_1d,
                                  np.maximum(high_1d - close_1d.shift(1),
                                             close_1d.shift(1) - low_1d))).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr_14 / np.log10(14) / (max_hh_14 - min_ll_14))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 12h volume spike: >1.5x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # EMA34 needs 34 bars, Williams %R needs 14, chop needs 14, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(williams_r_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Regime filter: Choppiness Index
        chop_value = chop_aligned[i]
        is_ranging = chop_value > 61.8  # Mean revert regime
        is_trending = chop_value < 38.2  # Trend follow regime
        
        # Williams %R levels
        wr_value = williams_r_aligned[i]
        oversold = wr_value < -80  # Oversold condition
        overbought = wr_value > -20  # Overbought condition
        
        # Trend filter: 1d EMA34 direction
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic based on regime
        long_entry = False
        short_entry = False
        
        if is_ranging:
            # In ranging markets: mean revert from Williams %R extremes
            long_entry = oversold and vol_confirm
            short_entry = overbought and vol_confirm
        elif is_trending:
            # In trending markets: follow 1d EMA34 trend with pullbacks to Williams %R extremes
            long_entry = price_above_ema and oversold and vol_confirm
            short_entry = price_below_ema and overbought and vol_confirm
        
        # Exit logic: opposite Williams %R extreme or regime change
        long_exit = wr_value > -20  # Exit long when overbought
        short_exit = wr_value < -80  # Exit short when oversold
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals