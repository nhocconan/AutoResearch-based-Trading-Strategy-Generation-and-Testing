# 2025-05-29
# Hypothesis: 4h Donchian(20) breakout with 1d trend filter, volume confirmation, and ATR volatility filter
# Donchian breakouts capture momentum moves; 1d EMA filter ensures trend alignment;
# Volume > 1.5x 20-period MA confirms breakout strength; ATR(14) > 0.5 * ATR(50) ensures sufficient volatility
# This combination aims for ~25-40 trades/year with clear entry/exit rules, suitable for both bull and bear markets
# Exit: price closes below/above the opposite Donchian band or ATR-based trailing stop (3x ATR from extreme)

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation (20-period MA)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    # ATR for volatility filter and trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr > 0.5 * atr_ma50  # Require sufficient volatility
    
    signals = np.zeros(n)
    position = 0
    # For trailing stop: track highest high since entry (long) or lowest low (short)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i]) or
            np.isnan(atr[i]) or np.isnan(atr_ma50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            continue
        
        # Update trailing stop levels
        if position == 1:  # Long position
            if i == 50 or position == 0:  # New entry
                highest_since_entry[i] = high[i]
            else:
                highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
        elif position == -1:  # Short position
            if i == 50 or position == 0:  # New entry
                lowest_since_entry[i] = low[i]
            else:
                lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
        else:  # Flat
            highest_since_entry[i] = 0
            lowest_since_entry[i] = 0
        
        if position == 0:
            # Long entry: price breaks above Donchian high + volume spike + price above 1d EMA34 + volatility filter
            if close[i] > donchian_high[i] and vol_spike[i] and close[i] > ema34_1d_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]
            # Short entry: price breaks below Donchian low + volume spike + price below 1d EMA34 + volatility filter
            elif close[i] < donchian_low[i] and vol_spike[i] and close[i] < ema34_1d_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
                lowest_since_entry[i] = low[i]
        else:
            # Exit conditions
            exit_signal = False
            if position == 1:  # Long position
                # Exit 1: price closes below Donchian low
                if close[i] < donchian_low[i]:
                    exit_signal = True
                # Exit 2: ATR-based trailing stop (3x ATR from highest high since entry)
                elif highest_since_entry[i] > 0 and close[i] < highest_since_entry[i] - 3.0 * atr[i]:
                    exit_signal = True
            else:  # Short position
                # Exit 1: price closes above Donchian high
                if close[i] > donchian_high[i]:
                    exit_signal = True
                # Exit 2: ATR-based trailing stop (3x ATR from lowest low since entry)
                elif lowest_since_entry[i] > 0 and close[i] > lowest_since_entry[i] + 3.0 * atr[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian_Breakout_1dEMA34_Volume_ATRFilter_v1"
timeframe = "4h"
leverage = 1.0