#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 1d Volume Spike + ATR Trailing Stop
# - Williams %R(14) identifies overbought/oversold levels: > -20 = overbought, < -80 = oversold
# - Trend filter: price > EMA(50) for long, price < EMA(50) for short to avoid counter-trend entries
# - 1d volume spike (>2.0x 20-day average) confirms institutional participation
# - ATR(14) trailing stop (2.5x) manages risk and adapts to volatility
# - Discrete position sizing (0.25) minimizes fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to avoid fee drag
# - Works in bull/bear: %R catches reversals in ranging markets, EMA filter avoids whipsaws in trends,
#   volume avoids false signals, ATR stop controls drawdown

name = "4h_1d_williamsr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d ATR volume for confirmation (14-period ATR)
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
    
    # Pre-compute 4h Williams %R (14-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    highest_high_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_4h) / (highest_high_14 - lowest_low_14)
    
    # Pre-compute 4h EMA(50) for trend filter
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Pre-compute 4h ATR for trailing stop (14-period)
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
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
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50[i]) or np.isnan(atr[i]) or 
            np.isnan(atr_volume_ma_aligned[i])):
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
        
        # Volume confirmation: current 1d ATR volume > 2.0x 20-day average
        volume_confirm = atr_volume_1d_aligned[i] > 2.0 * atr_volume_ma_aligned[i]
        
        # Williams %R conditions
        wr_value = williams_r[i]
        close_price = close_4h[i]
        ema_value = ema_50[i]
        
        # Oversold conditions for long: Williams %R < -80 AND price > EMA(50) (uptrend filter)
        oversold = wr_value < -80
        uptrend_filter = close_price > ema_value
        
        # Overbought conditions for short: Williams %R > -20 AND price < EMA(50) (downtrend filter)
        overbought = wr_value > -20
        downtrend_filter = close_price < ema_value
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Oversold AND volume confirmation AND uptrend filter
            if oversold and volume_confirm and uptrend_filter:
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                highest_since_entry = prices['high'].iloc[i]
                signals[i] = 0.25
            # Short conditions: Overbought AND volume confirmation AND downtrend filter
            elif overbought and volume_confirm and downtrend_filter:
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
                # ATR trailing stop: exit when price drops 2.5*ATR from highest point
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 2.5 * atr[i]
                exit_condition = trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # ATR trailing stop: exit when price rises 2.5*ATR from lowest point
                trailing_stop = prices['close'].iloc[i] > lowest_since_entry + 2.5 * atr[i]
                exit_condition = trailing_stop
            
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