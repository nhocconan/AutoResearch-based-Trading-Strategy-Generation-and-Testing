#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h strategy using 1w/1d Camarilla pivot levels with volume confirmation
    # Works in both bull and bear: Pivots provide mean-reversion levels, volume confirms breakouts,
    # 1w trend filter (EMA50) avoids counter-trend trades. Discrete sizing (0.25) minimizes fee drag.
    # Target: 12-30 trades/year to stay within 12h optimal range.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Get 1d data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Camarilla: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    # H3 = C + 1.1*(H-L)/4, L3 = C - 1.1*(H-L)/4
    # H2 = C + 1.1*(H-L)/6, L2 = C - 1.1*(H-L)/6
    # H1 = C + 1.1*(H-L)/12, L1 = C - 1.1*(H-L)/12
    # Pivot = (H+L+C)/3
    camarilla_h4 = np.zeros_like(close_1d)
    camarilla_l4 = np.zeros_like(close_1d)
    camarilla_h3 = np.zeros_like(close_1d)
    camarilla_l3 = np.zeros_like(close_1d)
    camarilla_h2 = np.zeros_like(close_1d)
    camarilla_l2 = np.zeros_like(close_1d)
    camarilla_h1 = np.zeros_like(close_1d)
    camarilla_l1 = np.zeros_like(close_1d)
    camarilla_pivot = np.zeros_like(close_1d)
    
    for i in range(1, len(high_1d)):
        # Use previous day's OHLC
        phigh = high_1d[i-1]
        plow = low_1d[i-1]
        pclose = close_1d[i-1]
        
        camarilla_pivot[i] = (phigh + plow + pclose) / 3
        camarilla_h1[i] = pclose + 1.1 * (phigh - plow) / 12
        camarilla_l1[i] = pclose - 1.1 * (phigh - plow) / 12
        camarilla_h2[i] = pclose + 1.1 * (phigh - plow) / 6
        camarilla_l2[i] = pclose - 1.1 * (phigh - plow) / 6
        camarilla_h3[i] = pclose + 1.1 * (phigh - plow) / 4
        camarilla_l3[i] = pclose - 1.1 * (phigh - plow) / 4
        camarilla_h4[i] = pclose + 1.1 * (phigh - plow) / 2
        camarilla_l4[i] = pclose - 1.1 * (phigh - plow) / 2
    
    # Get 1d volume for confirmation (20-period average)
    vol_avg_20_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all HTF indicators to 12h primary timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.2x 20-period EMA
        idx_1d = i // (24 * 2)  # 1d bars in 12h timeframe (2 bars per day)
        if idx_1d >= len(volume_1d):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_1d[idx_1d] > 1.2 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: price above/below 1w EMA50
        uptrend = close[i] > ema50_1w_aligned[i]
        downtrend = close[i] < ema50_1w_aligned[i]
        
        # Entry conditions: Camarilla level touch + trend + volume
        # Long: touch L3/L4 in uptrend with volume
        enter_long = (
            (close[i] <= camarilla_l3_aligned[i] * 1.001 and close[i] >= camarilla_l4_aligned[i] * 0.999) or
            (close[i] <= camarilla_l4_aligned[i] * 1.001 and close[i] >= camarilla_l4_aligned[i] * 0.999)
        ) and uptrend and volume_confirmed
        
        # Short: touch H3/H4 in downtrend with volume
        enter_short = (
            (close[i] >= camarilla_h3_aligned[i] * 0.999 and close[i] <= camarilla_h4_aligned[i] * 1.001) or
            (close[i] >= camarilla_h4_aligned[i] * 0.999 and close[i] <= camarilla_h4_aligned[i] * 1.001)
        ) and downtrend and volume_confirmed
        
        # Stoploss: 1.5x ATR based on 12h true range (simplified using Camarilla width)
        camarilla_width = camarilla_h4_aligned[i] - camarilla_l4_aligned[i]
        stop_distance = camarilla_width * 0.15  # 15% of Camarilla width
        
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - stop_distance
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + stop_distance
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
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

name = "12h_1w_1d_camarilla_pivot_volume_trend_v1"
timeframe = "12h"
leverage = 1.0