#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Use 1h timeframe for entry timing precision
# - Use 4h Donchian(20) trend for signal direction (long only above 4h Donchian mid, short only below)
# - Use 1d Camarilla levels (H3/L3) for high-probability breakout entries
# - Volume confirmation: 1h volume > 1.3x 20-period 1h volume SMA
# - Session filter: 08-20 UTC to avoid low-liquidity hours
# - Position sizing: 0.20 discrete level
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - This combines HTF structure (4h trend + 1d pivots) with LTF timing (1h entries) to reduce whipsaw
# - Works in both bull and bear markets by following HTF direction and using mean-reversion pivots for entries

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h Donchian(20) for trend filter
    donchian_period = 20
    if len(df_4h) >= donchian_period:
        donchian_high_4h = pd.Series(df_4h['high'].values).rolling(window=donchian_period, min_periods=donchian_period).max().values
        donchian_low_4h = pd.Series(df_4h['low'].values).rolling(window=donchian_period, min_periods=donchian_period).min().values
        donchian_mid_4h = (donchian_high_4h + donchian_low_4h) / 2.0
        # Align to 1h timeframe (completed 4h bars only)
        donchian_mid_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_4h)
    else:
        donchian_mid_4h_aligned = np.full(n, np.nan)
    
    # Calculate 1d Camarilla levels (H3/L3)
    if len(df_1d) >= 2:
        # Camarilla: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        camarilla_h3_1d = close_1d + 1.1 * (high_1d - low_1d) / 2.0
        camarilla_l3_1d = close_1d - 1.1 * (high_1d - low_1d) / 2.0
        # Use previous day's levels (completed 1d bars only)
        camarilla_h3_1d_prev = np.roll(camarilla_h3_1d, 1)
        camarilla_l3_1d_prev = np.roll(camarilla_l3_1d, 1)
        camarilla_h3_1d_prev[0] = np.nan
        camarilla_l3_1d_prev[0] = np.nan
        # Align to 1h timeframe (completed 1d bars only)
        camarilla_h3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d_prev)
        camarilla_l3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d_prev)
    else:
        camarilla_h3_1d_aligned = np.full(n, np.nan)
        camarilla_l3_1d_aligned = np.full(n, np.nan)
    
    # Calculate 1h volume SMA(20) for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_mid_4h_aligned[i]) or
            np.isnan(camarilla_h3_1d_aligned[i]) or
            np.isnan(camarilla_l3_1d_aligned[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1h volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > 1.3 * volume_sma_20[i]
        
        # Trend filter: price relative to 4h Donchian mid
        above_4h_trend = close[i] > donchian_mid_4h_aligned[i]
        below_4h_trend = close[i] < donchian_mid_4h_aligned[i]
        
        if position == 0:  # Flat - look for entry
            # Long: price breaks above 1d H3, above 4h trend, with volume
            if (close[i] > camarilla_h3_1d_aligned[i] and 
                above_4h_trend and vol_confirm):
                position = 1
                signals[i] = 0.20
                entry_price[i] = close[i]
            # Short: price breaks below 1d L3, below 4h trend, with volume
            elif (close[i] < camarilla_l3_1d_aligned[i] and 
                  below_4h_trend and vol_confirm):
                position = -1
                signals[i] = -0.20
                entry_price[i] = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit on Donchian midpoint reversion (4h) or opposite Camarilla break
            exit_condition = (close[i] < donchian_mid_4h_aligned[i]) or \
                           (close[i] < camarilla_l3_1d_aligned[i])
            if exit_condition:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.20
        else:  # position == -1 (Short position) - look for exit
            # Exit on Donchian midpoint reversion (4h) or opposite Camarilla break
            exit_condition = (close[i] > donchian_mid_4h_aligned[i]) or \
                           (close[i] > camarilla_h3_1d_aligned[i])
            if exit_condition:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.20
    
    return signals