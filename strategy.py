#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d EMA50 trend filter and volume confirmation
# - Uses Camarilla pivot levels from 1d (structure-based support/resistance)
# - 1d EMA50 trend filter ensures trades align with intermediate trend (works in bull/bear)
# - Volume confirmation: current volume > 1.8x 20-period average to filter weak breakouts
# - Exit: touch of opposite Camarilla level or ATR-based stoploss (1.5*ATR)
# - Position size: 0.25 (25% of capital) to balance risk and minimize fee drag
# - Target: 20-50 trades/year on 4h (80-200 total over 4 years) to stay within trade limits
# - Works in bull/bear: EMA50 adapts to regime changes, volume reduces false signals, Camarilla provides objective levels

name = "4h_1d_camarilla_breakout_volume_v2"
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
    
    # Pre-compute 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute 1d Camarilla pivot levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_h3 = []
    camarilla_l3 = []
    camarilla_h4 = []
    camarilla_l4 = []
    
    for i in range(len(close_1d)):
        if i == 0:
            camarilla_h3.append(np.nan)
            camarilla_l3.append(np.nan)
            camarilla_h4.append(np.nan)
            camarilla_l4.append(np.nan)
        else:
            high = high_1d[i-1]
            low = low_1d[i-1]
            close = close_1d_prev[i-1]
            range_val = high - low
            
            camarilla_h3.append(close + range_val * 1.1 / 4)
            camarilla_l3.append(close - range_val * 1.1 / 4)
            camarilla_h4.append(close + range_val * 1.1 / 2)
            camarilla_l4.append(close - range_val * 1.1 / 2)
    
    camarilla_h3 = np.array(camarilla_h3)
    camarilla_l3 = np.array(camarilla_l3)
    camarilla_h4 = np.array(camarilla_h4)
    camarilla_l4 = np.array(camarilla_l4)
    
    # Align Camarilla levels to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute 4h volume average (20-period)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ATR for stoploss (14-period)
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
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(trend_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirm = volume[i] > 1.8 * vol_ma_20[i]
        
        # Get current 1d close for trend filter (aligned)
        close_1d_current = df_1d['close'].values
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_current)
        
        # 1d trend filter: price > EMA50 = bullish, price < EMA50 = bearish
        bullish_trend = not np.isnan(close_1d_aligned[i]) and not np.isnan(trend_aligned[i]) and \
                        close_1d_aligned[i] > trend_aligned[i]
        bearish_trend = not np.isnan(close_1d_aligned[i]) and not np.isnan(trend_aligned[i]) and \
                        close_1d_aligned[i] < trend_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price > Camarilla H3 AND bullish trend AND volume confirmation
            if prices['close'].iloc[i] > h3_aligned[i] and bullish_trend and volume_confirm:
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                signals[i] = 0.25
            # Short conditions: price < Camarilla L3 AND bearish trend AND volume confirmation
            elif prices['close'].iloc[i] < l3_aligned[i] and bearish_trend and volume_confirm:
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or stoploss
            # Exit conditions: price touches opposite Camarilla level
            exit_long = prices['close'].iloc[i] < l3_aligned[i]   # Price breaks below Camarilla L3 (exit long)
            exit_short = prices['close'].iloc[i] > h3_aligned[i]  # Price breaks above Camarilla H3 (exit short)
            
            # Stoploss conditions: ATR-based (1.5 * ATR)
            if position == 1:
                stoploss_hit = prices['close'].iloc[i] < entry_price - 1.5 * atr[i]
            else:  # position == -1
                stoploss_hit = prices['close'].iloc[i] > entry_price + 1.5 * atr[i]
            
            exit_condition = exit_long or exit_short or stoploss_hit
            
            if exit_condition:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals