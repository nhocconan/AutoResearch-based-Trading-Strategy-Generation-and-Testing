#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR volume filter and 1w trend filter
# - Donchian breakout captures momentum moves with clear structure
# - 1d ATR volume filter ensures breakouts occur with institutional participation
# - 1w EMA trend filter ensures alignment with higher timeframe direction
# - ATR(14) trailing stop (2.0x) manages risk without whipsaw
# - Position size: 0.25 discrete to minimize fee churn
# - Target: 20-30 trades/year (80-120 total over 4 years) to avoid fee drag
# - Works in bull/bear: Donchian adapts to volatility, volume filter avoids false breakouts, trend filter avoids counter-trend trades

name = "4h_1d_1w_donchian_volume_trend_v1"
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
    
    # Pre-compute 1d ATR for volume filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = np.nan
    tr2_1d[0] = np.nan
    tr3_1d[0] = np.nan
    tr_1d = np.maximum.reduce([tr1_1d, tr2_1d, tr3_1d])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 1d ATR volume: volume / ATR (normalizes volume by volatility)
    atr_volume_1d = volume_1d / atr_1d
    atr_volume_ma_20_1d = pd.Series(atr_volume_1d).rolling(window=20, min_periods=20).mean().values
    atr_volume_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_volume_ma_20_1d)
    
    # Pre-compute 1w EMA for trend filter (21-period)
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Pre-compute 4h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h ATR for trailing stop (14-period)
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
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_volume_ma_aligned[i]) or np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(atr[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current 1d ATR volume for filter (aligned)
        atr_volume_1d_current = atr_volume_1d
        atr_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_volume_1d_current)
        
        # Volume confirmation: current 1d ATR volume > 1.5x 20-day average
        volume_confirm = atr_volume_1d_aligned[i] > 1.5 * atr_volume_ma_aligned[i]
        
        # Trend filter: price above/below 1w EMA
        trend_long = prices['close'].iloc[i] > ema_21_1w_aligned[i]
        trend_short = prices['close'].iloc[i] < ema_21_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price > Donchian high AND volume confirmation AND uptrend
            if prices['close'].iloc[i] > donchian_high[i] and volume_confirm and trend_long:
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                highest_since_entry = prices['high'].iloc[i]
                signals[i] = 0.25
            # Short conditions: price < Donchian low AND volume confirmation AND downtrend
            elif prices['close'].iloc[i] < donchian_low[i] and volume_confirm and trend_short:
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
                # Exit conditions: price < Donchian low (stop loss) OR ATR trailing stop
                exit_long = prices['close'].iloc[i] < donchian_low[i]  # Stop loss at Donchian low
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 2.0 * atr[i]
                exit_condition = exit_long or trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # Exit conditions: price > Donchian high (stop loss) OR ATR trailing stop
                exit_short = prices['close'].iloc[i] > donchian_high[i]  # Stop loss at Donchian high
                trailing_stop = prices['close'].iloc[i] > lowest_since_entry + 2.0 * atr[i]
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