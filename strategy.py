#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Camarilla Pivot Breakout + Volume Spike + ATR Trailing Stop
# - Weekly Camarilla pivot levels (H3/L3) act as strong institutional support/resistance
# - Breakout above H3 or below L3 with daily volume spike confirms institutional participation
# - ATR(14) trailing stop (2.0x) manages risk and adapts to volatility
# - Works in bull/bear: Weekly Camarilla levels adapt to price action, volume filter avoids false breakouts
# - Target: 10-25 trades/year (40-100 total over 4 years) to minimize fee drag

name = "1d_weekly_camarilla_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop: weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute weekly ATR volume for confirmation (14-period ATR)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    tr1_1w = high_1w - low_1w
    tr2_1w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_1w = np.abs(low_1w - np.roll(close_1w, 1))
    tr1_1w[0] = np.nan
    tr2_1w[0] = np.nan
    tr3_1w[0] = np.nan
    tr_1w = np.maximum.reduce([tr1_1w, tr2_1w, tr3_1w])
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # Weekly ATR volume: volume / ATR (normalizes volume by volatility)
    atr_volume_1w = volume_1w / atr_1w
    atr_volume_ma_20_1w = pd.Series(atr_volume_1w).rolling(window=20, min_periods=20).mean().values
    atr_volume_ma_aligned = align_htf_to_ltf(prices, df_1w, atr_volume_ma_20_1w)
    
    # Pre-compute weekly Camarilla pivot levels from previous week
    # Need weekly OHLC for pivot calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla levels: H3/L3 from previous week
    # H3 = close + 1.1*(high-low)/2
    # L3 = close - 1.1*(high-low)/2
    camarilla_h3 = close_1w + 1.1 * (high_1w - low_1w) / 2.0
    camarilla_l3 = close_1w - 1.1 * (high_1w - low_1w) / 2.0
    
    # Align Camarilla levels to 1d timeframe (use previous week's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    # Pre-compute 1d ATR for trailing stop (14-period)
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
        
        # Get current weekly ATR volume for filter (aligned)
        atr_volume_1w_current = atr_volume_1w
        atr_volume_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_volume_1w_current)
        
        # Volume confirmation: current weekly ATR volume > 1.8x 20-week average
        volume_confirm = atr_volume_1w_aligned[i] > 1.8 * atr_volume_ma_aligned[i]
        
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