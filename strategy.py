#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h volume confirmation and ATR filter
# - Donchian(20) on 4h: breakout above upper band = long, below lower band = short
# - Volume confirmation: current 4h volume > 1.5x 20-period EMA
# - ATR filter: only trade when ATR(14) > 0.5 * ATR(50) (avoid low volatility)
# - Exit: opposite Donchian breakout or ATR trailing stop (2.5 * ATR from extreme)
# - Position sizing: 0.25 discrete level
# - Targets ~20-50 trades/year on 4h timeframe. Uses Donchian for structure,
#   volume confirmation avoids fakeouts, ATR filter ensures sufficient volatility.
#   Works in bull/bear: breakouts capture strong moves, volume/ATR filters reduce whipsaws.

name = "4h_12h_donchian_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Donchian channels (20-period) on 4h
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR for volatility filter and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period EMA
    volume_ema_20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    vol_confirm = volume > 1.5 * volume_ema_20
    
    # ATR filter: only trade when short-term ATR > 0.5 * long-term ATR
    atr_filter = atr_14 > 0.5 * atr_50
    
    # Calculate 12h volume EMA for HTF confirmation
    volume_ema_20_12h = pd.Series(volume_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ema_20_12h)
    
    # Track extremes for trailing stop
    long_extreme = 0.0
    short_extreme = 0.0
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(atr_14[i]) or np.isnan(atr_50[i]) or 
            np.isnan(volume_ema_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # HTF volume confirmation: 12h volume > 1.2x its 20-period EMA
        vol_12h_current = align_htf_to_ltf(prices, df_12h, volume_12h)
        vol_confirm_12h = vol_12h_current[i] > 1.2 * volume_ema_20_12h_aligned[i]
        
        # Entry conditions
        long_entry = (close[i] > highest_high_20[i] and 
                     vol_confirm[i] and vol_confirm_12h and atr_filter[i])
        short_entry = (close[i] < lowest_low_20[i] and 
                      vol_confirm[i] and vol_confirm_12h and atr_filter[i])
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
                long_extreme = high[i]  # Initialize long extreme
            elif short_entry:
                position = -1
                signals[i] = -0.25
                short_extreme = low[i]  # Initialize short extreme
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Update extremes
            if position == 1:  # Long position
                long_extreme = max(long_extreme, high[i])
                # ATR trailing stop: exit if price drops 2.5*ATR from extreme
                if close[i] < long_extreme - 2.5 * atr_14[i]:
                    position = 0
                    signals[i] = 0.0
                # Opposite Donchian breakout
                elif close[i] < lowest_low_20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                short_extreme = min(short_extreme, low[i])
                # ATR trailing stop: exit if price rises 2.5*ATR from extreme
                if close[i] > short_extreme + 2.5 * atr_14[i]:
                    position = 0
                    signals[i] = 0.0
                # Opposite Donchian breakout
                elif close[i] > highest_high_20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals