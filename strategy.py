#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and session filter (08-20 UTC)
    # Works in both bull and bear: Camarilla captures intraday reversals, 4h trend filters false breakouts,
    # session filter reduces noise during low-volume hours, volume confirmation ensures momentum
    # Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values if 'volume' in df_4h.columns else np.ones(len(df_4h))
    
    # Calculate 4h Camarilla levels (H3, L3, H4, L4)
    camarilla_h3 = close_4h + (high_4h - low_4h) * 1.1 / 4
    camarilla_l3 = close_4h - (high_4h - low_4h) * 1.1 / 4
    camarilla_h4 = close_4h + (high_4h - low_4h) * 1.1 / 2
    camarilla_l4 = close_4h - (high_4h - low_4h) * 1.1 / 2
    
    # Get 4h EMA(20) for trend filter
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Get 4h volume for confirmation (20-period average)
    vol_avg_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all 4h indicators to 1h primary timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4)
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    vol_avg_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% position size
    atr_multiplier = 1.5  # ATR stoploss multiplier
    
    # Calculate 1h ATR for stoploss
    tr = np.maximum(np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1])), np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr_1h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(ema_20_4h_aligned[i]) or
            np.isnan(vol_avg_20_4h_aligned[i]) or
            np.isnan(atr_1h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.2x 20-period average
        idx_4h = i // 4
        if idx_4h >= len(volume_4h):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_4h[idx_4h] > 1.2 * vol_avg_20_4h_aligned[i]
        
        # Trend direction from 4h EMA(20)
        trend_up = close[i] > ema_20_4h_aligned[i]
        trend_down = close[i] < ema_20_4h_aligned[i]
        
        # Entry conditions: Camarilla H3/L3 break + trend + volume + session
        enter_long = (close[i] > camarilla_h3_aligned[i]) and trend_up and volume_confirmed
        enter_short = (close[i] < camarilla_l3_aligned[i]) and trend_down and volume_confirmed
        
        # Stoploss conditions
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - atr_multiplier * atr_1h[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + atr_multiplier * atr_1h[i]
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]  # record entry price at close (filled next bar open)
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]  # record entry price at close (filled next bar open)
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "1h_4h_camarilla_pivot_breakout_trend_volume_session_v1"
timeframe = "1h"
leverage = 1.0