#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot Breakout + 12h Volume Spike + ATR Trailing Stop
# - 4h Camarilla pivot levels (H3/L3) act as institutional support/resistance
# - Breakout above H3 or below L3 with 12h volume spike confirms institutional participation
# - ATR(14) trailing stop (2.0x) manages risk and adapts to volatility
# - Discrete position sizing (0.25) minimizes fee churn
# - Target: 15-35 trades/year (60-140 total over 4 years) to avoid fee drag
# - Works in bull/bear: Camarilla levels adapt to price action, volume filter avoids false breakouts, ATR stop controls drawdown

name = "4h_12h_camarilla_breakout_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h ATR volume for confirmation (14-period ATR)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr1_12h[0] = np.nan
    tr2_12h[0] = np.nan
    tr3_12h[0] = np.nan
    tr_12h = np.maximum.reduce([tr1_12h, tr2_12h, tr3_12h])
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # 12h ATR volume: volume / ATR (normalizes volume by volatility)
    atr_volume_12h = volume_12h / atr_12h
    atr_volume_ma_20_12h = pd.Series(atr_volume_12h).rolling(window=20, min_periods=20).mean().values
    atr_volume_ma_aligned = align_htf_to_ltf(prices, df_12h, atr_volume_ma_20_12h)
    
    # Pre-compute 4h Camarilla pivot levels from previous day
    # Need daily OHLC for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day: based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4/H3/L3/L4 from previous day
    # H3 = close + 1.1*(high-low)/2
    # L3 = close - 1.1*(high-low)/2
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 2.0
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 2.0
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
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
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(atr_volume_ma_aligned[i]) or np.isnan(atr[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current 12h ATR volume for filter (aligned)
        atr_volume_12h_current = atr_volume_12h
        atr_volume_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_volume_12h_current)
        
        # Volume confirmation: current 12h ATR volume > 1.8x 20-day average
        volume_confirm = atr_volume_12h_aligned[i] > 1.8 * atr_volume_ma_aligned[i]
        
        # Price levels for breakout
        current_close = prices['close'].iloc[i]
        camarilla_h3_level = camarilla_h3_aligned[i]
        camarilla_l3_level = camarilla_l3_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Close above H3 AND volume confirmation
            if current_close > camarilla_h3_level and volume_confirm:
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else current_close
                highest_since_entry = prices['high'].iloc[i]
                signals[i] = 0.25
            # Short conditions: Close below L3 AND volume confirmation
            elif current_close < camarilla_l3_level and volume_confirm:
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else current_close
                lowest_since_entry = prices['low'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or trailing stop
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                # ATR trailing stop: exit when price drops 2.0*ATR from highest point
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 2.0 * atr[i]
                exit_condition = trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # ATR trailing stop: exit when price rises 2.0*ATR from lowest point
                trailing_stop = prices['close'].iloc[i] > lowest_since_entry + 2.0 * atr[i]
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