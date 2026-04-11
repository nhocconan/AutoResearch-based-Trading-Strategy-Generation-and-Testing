#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's close, high, low
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]  # first value
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Camarilla levels calculation
    range_ = prev_high - prev_low
    camarilla_H4 = prev_close + 1.1 * range_ / 2  # resistance
    camarilla_L4 = prev_close - 1.1 * range_ / 2  # support
    
    # Align to 12h timeframe
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    
    # Volume confirmation: volume > 1.3x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # ATR for stop loss and trend filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Trend filter: ATR > 0.7 * 50-period ATR average (avoid choppy markets)
    atr_ma_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    trending_market = atr > 0.7 * atr_ma_50
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_H4_aligned[i]) or np.isnan(camarilla_L4_aligned[i]) or
            np.isnan(vol_ma_30[i]) or np.isnan(atr[i]) or np.isnan(trending_market[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        H4 = camarilla_H4_aligned[i]
        L4 = camarilla_L4_aligned[i]
        atr_val = atr[i]
        trending = trending_market[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.3 * vol_ma_30[i]
        
        # Entry signals - only in trending markets to avoid whipsaws
        long_signal = False
        short_signal = False
        
        # Long: price breaks above Camarilla H4 with volume and trending
        if price_high > H4 and volume_confirmed and trending:
            long_signal = True
        
        # Short: price breaks below Camarilla L4 with volume and trending
        if price_low < L4 and volume_confirmed and trending:
            short_signal = True
        
        # Exit conditions
        # Stop loss conditions
        stop_long = position == 1 and price_low < (entry_price - 1.5 * atr_val)
        stop_short = position == -1 and price_high > (entry_price + 1.5 * atr_val)
        
        # Exit when price returns to Camarilla pivot (close level)
        camarilla_pivot = (prev_close[i] + prev_high[i] + prev_low[i]) / 3
        camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)[i]
        exit_long = position == 1 and price_close < camarilla_pivot_aligned
        exit_short = position == -1 and price_close > camarilla_pivot_aligned
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.25
        elif position == 1 and (exit_long or stop_long):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (exit_short or stop_short):
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 12h Camarilla breakout strategy with volume confirmation and trend filter.
# Enters long when price breaks above daily Camarilla H4 level with volume confirmation (>1.3x avg volume) in trending markets.
# Enters short when price breaks below daily Camarilla L4 level with volume confirmation and trending.
# Uses Camarilla levels from daily timeframe for institutional-grade support/resistance.
# Volume confirmation ensures institutional participation, trend filter avoids whipsaws in sideways markets.
# Exits when price returns to Camarilla pivot point or ATR stop loss (1.5x) is hit.
# Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year).
# Works in both bull and bear markets by trading breakouts in either direction with trend filter.