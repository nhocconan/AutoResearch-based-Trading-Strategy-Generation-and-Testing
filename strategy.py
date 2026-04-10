#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and 1w ADX trend filter
# - Camarilla levels from 1d provide precise intraday support/resistance for breakouts
# - 1d volume spike (>2x 20-period average) confirms institutional participation
# - 1w ADX > 25 ensures we only trade in trending markets, avoiding chop
# - ATR(14) trailing stop (2.5x) manages risk with volatility adaptation
# - Position size: 0.25 discrete to minimize fee churn
# - Target: 20-30 trades/year (80-120 total over 4 years) to avoid fee drag
# - Works in bull/bear: Camarilla adapts to volatility, volume filter avoids false breakouts, ADX filter avoids ranging markets

name = "4h_1d_1w_camarilla_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for today (using yesterday's data)
    # Camarilla: H4 = Close + 1.1*(High-Low)/2, L4 = Close - 1.1*(High-Low)/2
    # We use H3/L3 for breakouts: H3 = Close + 1.1*(High-Low)/4, L3 = Close - 1.1*(High-Low)/4
    rng_1d = high_1d - low_1d
    camarilla_h3_1d = close_1d + 1.1 * rng_1d / 4
    camarilla_l3_1d = close_1d - 1.1 * rng_1d / 4
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    # Pre-compute 1d volume spike filter (volume > 2x 20-period MA)
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * volume_ma_20_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Pre-compute 1w ADX for trend filter (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr_1w = np.maximum.reduce([tr1, tr2, tr3])
    
    # Calculate Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (using Wilder's smoothing = EMA with alpha=1/period)
    atr_1w = pd.Series(tr_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_ema = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_ema = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus_ema / (atr_1w + 1e-10)
    di_minus = 100 * dm_minus_ema / (atr_1w + 1e-10)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_1w = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Pre-compute 4h ATR for trailing stop (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(adx_1w_aligned[i]) or 
            np.isnan(atr[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation: 1d volume spike
        volume_confirm = volume_spike_aligned[i] > 0.5  # Boolean as float
        
        # Trend filter: 1w ADX > 25 indicates trending market
        trend_filter = adx_1w_aligned[i] > 25.0
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price > Camarilla H3 AND volume confirmation AND trending market
            if (prices['close'].iloc[i] > camarilla_h3_aligned[i] and 
                volume_confirm and trend_filter):
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                highest_since_entry = prices['high'].iloc[i]
                signals[i] = 0.25
            # Short conditions: price < Camarilla L3 AND volume confirmation AND trending market
            elif (prices['close'].iloc[i] < camarilla_l3_aligned[i] and 
                  volume_confirm and trend_filter):
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                lowest_since_entry = prices['low'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or trailing stop
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                # Exit conditions: price < Camarilla L3 (stop loss) OR ATR trailing stop
                exit_long = prices['close'].iloc[i] < camarilla_l3_aligned[i]  # Stop loss at Camarilla L3
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 2.5 * atr[i]
                exit_condition = exit_long or trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # Exit conditions: price > Camarilla H3 (stop loss) OR ATR trailing stop
                exit_short = prices['close'].iloc[i] > camarilla_h3_aligned[i]  # Stop loss at Camarilla H3
                trailing_stop = prices['close'].iloc[i] > lowest_since_entry + 2.5 * atr[i]
                exit_condition = exit_short or trailing_stop
            
            if exit_condition:
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals