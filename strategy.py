#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Choppiness Index (14) regime filter with 1-day Donchian(20) breakout and volume confirmation
# In trending markets (CHOP < 38.2): trade breakouts in direction of trend (price > EMA50 for long, < EMA50 for short)
# In ranging markets (CHOP > 61.8): fade extremes (short at Donchian high, long at Donchian low)
# Volume confirmation: current volume > 1.5x 20-period average
# Stoploss: 2.0 * ATR(14)
# Position size: 0.25
# Designed to work in both bull and bear regimes by adapting to market conditions

name = "12h_chop_regime_donchian20_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Choppiness Index (14) - measures ranging vs trending markets
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(tr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    range_hl = hh - ll
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop = 100 * np.log10(tr_sum / range_hl) / np.log10(14)
    chop = np.where(np.isnan(chop), 50.0, chop)  # neutral when undefined
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # EMA(50) for trend direction
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 20-period Donchian channels (12h)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(chop_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        chop_val = chop_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_ma = volume_ma[i]
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit conditions based on regime
            elif chop_val < 38.2:  # trending - exit if trend reverses
                if close[i] < ema_val:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                else:
                    signals[i] = 0.25
            elif chop_val > 61.8:  # ranging - exit at opposite Donchian
                if close[i] > lowest_low[i]:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                else:
                    signals[i] = 0.25
            else:  # transition - use Donchian exit
                if close[i] < lowest_low[i]:
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
            # Exit conditions based on regime
            elif chop_val < 38.2:  # trending - exit if trend reverses
                if close[i] > ema_val:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                else:
                    signals[i] = -0.25
            elif chop_val > 61.8:  # ranging - exit at opposite Donchian
                if close[i] < highest_high[i]:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                else:
                    signals[i] = -0.25
            else:  # transition - use Donchian exit
                if close[i] > highest_high[i]:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                else:
                    signals[i] = -0.25
        else:
            # Look for entries based on regime
            vol_filter = volume[i] > 1.5 * vol_ma
            
            if chop_val < 38.2:  # trending market
                # Long: price > EMA50 and breaks above Donchian high
                if close[i] > ema_val and close[i] > highest_high[i] and vol_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price < EMA50 and breaks below Donchian low
                elif close[i] < ema_val and close[i] < lowest_low[i] and vol_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            elif chop_val > 61.8:  # ranging market
                # Long: price breaks below Donchian low (mean reversion)
                if close[i] < lowest_low[i] and vol_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price breaks above Donchian high (mean reversion)
                elif close[i] > highest_high[i] and vol_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:  # transition regime - use Donchian breakouts with EMA filter
                # Long: price above EMA50 and breaks above Donchian high
                if close[i] > ema_val and close[i] > highest_high[i] and vol_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price below EMA50 and breaks below Donchian low
                elif close[i] < ema_val and close[i] < lowest_low[i] and vol_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals
</think>