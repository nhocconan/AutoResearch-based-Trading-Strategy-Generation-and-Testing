#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams %R + 12-hour EMA trend filter + 1-day volume confirmation
# Long when Williams %R crosses above -50 (bullish momentum) + price > 12-hour EMA + volume > 1.5x 20-period average
# Short when Williams %R crosses below -50 (bearish momentum) + price < 12-hour EMA + volume > 1.5x 20-period average
# Exit when Williams %R crosses back below -20 (long) or above -80 (short)
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_williamsr_12h_ema_1d_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12-hour data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 1-day data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 12-hour EMA (20-period)
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 1-day volume average (20-period)
    volume_1d = df_1d['volume'].values
    volume_1d_s = pd.Series(volume_1d)
    volume_ma = volume_1d_s.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    # Williams %R (14-period) - calculated on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(williams_r[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R crosses below -20 (overbought)
            elif williams_r[i] < -20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R crosses above -80 (oversold)
            elif williams_r[i] > -80:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Williams %R crossing -50 with trend and volume filters
            # Volume filter: volume > 1.5x 20-period average
            volume_filter = volume[i] > 1.5 * volume_ma_aligned[i]
            # Trend filter: price > 12h EMA for long, price < 12h EMA for short
            trend_filter_long = close[i] > ema_12h_aligned[i]
            trend_filter_short = close[i] < ema_12h_aligned[i]
            
            # Williams %R just crossed above -50 (bullish momentum)
            wr_cross_up = williams_r[i] > -50 and williams_r[i-1] <= -50
            # Williams %R just crossed below -50 (bearish momentum)
            wr_cross_down = williams_r[i] < -50 and williams_r[i-1] >= -50
            
            # Long: Williams %R crosses above -50 + price > 12h EMA + volume filter
            if wr_cross_up and trend_filter_long and volume_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Williams %R crosses below -50 + price < 12h EMA + volume filter
            elif wr_cross_down and trend_filter_short and volume_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals