#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla breakout with 12h trend filter and volume confirmation
    # Camarilla levels from 1d pivot capture key support/resistance.
    # 12h EMA(50) filters false breakouts in choppy markets.
    # Volume confirmation ensures momentum behind breakouts.
    # Works in both bull/bear: trend filter adapts to regime, volume avoids low-conviction moves.
    # Target: 12-30 trades/year to minimize fee drag on 6h timeframe.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for Camarilla levels and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels (H3/L3 for fade, H4/L4 for breakout)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    daily_volume = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    camarilla_h4 = daily_close + (daily_high - daily_low) * 1.1 / 2
    camarilla_l4 = daily_close - (daily_high - daily_low) * 1.1 / 2
    camarilla_h3 = daily_close + (daily_high - daily_low) * 1.1 / 4
    camarilla_l3 = daily_close - (daily_high - daily_low) * 1.1 / 4
    
    # Get 12h data for trend filter (EMA 50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 12h volume for confirmation (20-period average)
    volume_12h = df_12h['volume'].values if 'volume' in df_12h.columns else np.ones(len(df_12h))
    vol_avg_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h primary timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    
    # Calculate 6h ATR for dynamic position sizing (optional, fixed size used for simplicity)
    tr = np.maximum(np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1])), np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr_6h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    atr_multiplier = 2.5  # ATR stoploss multiplier
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_avg_20_12h_aligned[i]) or
            np.isnan(atr_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        idx_12h = i // 2  # 2x 6h bars per 12h bar
        if idx_12h >= len(volume_12h):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_12h[idx_12h] > 1.5 * vol_avg_20_12h_aligned[i]
        
        # Trend direction from 12h EMA(50)
        trend_up = close[i] > ema_50_12h_aligned[i]
        trend_down = close[i] < ema_50_12h_aligned[i]
        
        # Entry conditions: Camarilla H4/L4 break + trend + volume
        # Only trade breakouts (not fades) for stronger momentum
        enter_long = (close[i] > camarilla_h4_aligned[i]) and trend_up and volume_confirmed
        enter_short = (close[i] < camarilla_l4_aligned[i]) and trend_down and volume_confirmed
        
        # Stoploss conditions
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - atr_multiplier * atr_6h[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + atr_multiplier * atr_6h[i]
        
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

name = "6h_12h_1d_camarilla_breakout_trend_volume_v1"
timeframe = "6h"
leverage = 1.0