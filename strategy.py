#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and 1w trend filter
# - Camarilla levels provide precise intraday support/resistance from 1d OHLC
# - Breakout above H3 or below L3 with volume confirmation captures institutional moves
# - 1w EMA trend filter ensures alignment with higher timeframe direction
# - ATR(14) trailing stop (2.0x) manages risk without whipsaw
# - Position size: 0.25 discrete to minimize fee churn
# - Target: 20-30 trades/year (80-120 total over 4 years) to avoid fee drag
# - Works in bull/bear: Camarilla adapts to volatility, volume filter avoids false breakouts, trend filter avoids counter-trend trades

name = "4h_1d_1w_camarilla_breakout_volume_v1"
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
    
    # Pre-compute 1d Camarilla levels (based on previous day OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Calculate Camarilla levels for each day (using previous day's data)
    camarilla_h3 = np.full_like(close_1d, np.nan)
    camarilla_l3 = np.full_like(close_1d, np.nan)
    camarilla_h4 = np.full_like(close_1d, np.nan)
    camarilla_l4 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        # Use previous day's OHLC to calculate today's levels
        high_y = high_1d[i-1]
        low_y = low_1d[i-1]
        close_y = close_1d[i-1]
        
        range_y = high_y - low_y
        camarilla_h3[i] = close_y + range_y * 1.1 / 4
        camarilla_l3[i] = close_y - range_y * 1.1 / 4
        camarilla_h4[i] = close_y + range_y * 1.1 / 2
        camarilla_l4[i] = close_y - range_y * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    h3_4h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_4h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_4h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_4h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute 1d volume confirmation (volume > 1.5x 20-day average)
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ratio_1d = volume_1d / volume_ma_20_1d
    volume_ratio_aligned = align_htf_to_ltf(prices, df_1d, volume_ratio_1d)
    
    # Pre-compute 1w EMA for trend filter (21-period)
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
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
        if (np.isnan(h3_4h[i]) or np.isnan(l3_4h[i]) or 
            np.isnan(volume_ratio_aligned[i]) or np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(atr[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        volume_confirm = volume_ratio_aligned[i] > 1.5
        
        # Trend filter: price above/below 1w EMA
        trend_long = prices['close'].iloc[i] > ema_21_1w_aligned[i]
        trend_short = prices['close'].iloc[i] < ema_21_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price > Camarilla H3 AND volume confirmation AND uptrend
            if prices['close'].iloc[i] > h3_4h[i] and volume_confirm and trend_long:
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                highest_since_entry = prices['high'].iloc[i]
                signals[i] = 0.25
            # Short conditions: price < Camarilla L3 AND volume confirmation AND downtrend
            elif prices['close'].iloc[i] < l3_4h[i] and volume_confirm and trend_short:
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
                exit_long = prices['close'].iloc[i] < l3_4h[i]  # Stop loss at Camarilla L3
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 2.0 * atr[i]
                exit_condition = exit_long or trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # Exit conditions: price > Camarilla H3 (stop loss) OR ATR trailing stop
                exit_short = prices['close'].iloc[i] > h3_4h[i]  # Stop loss at Camarilla H3
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