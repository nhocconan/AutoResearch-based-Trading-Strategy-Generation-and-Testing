#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1w trend filter
# - Donchian breakout on 4h: price > 20-bar high = long, price < 20-bar low = short
# - Volume confirmation: current 1d volume > 1.5x 20-bar EMA (avoid low-volume fakeouts)
# - Trend filter: price > 1w EMA(50) for longs, price < 1w EMA(50) for shorts
# - Exit: ATR-based trailing stop (3x ATR) or opposite Donchian breakout
# - Position sizing: 0.25 discrete level to balance risk and return
# - Targets ~20-40 trades/year on 4h timeframe. Donchian captures trends,
#   volume confirmation reduces whipsaws, 1w EMA ensures alignment with higher timeframe trend.
#   Works in bull/bear: breakouts catch momentum, volume/trend filters improve win rate.

name = "4h_1d_1w_donchian_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    close_1w = df_1w['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Calculate Donchian channels on 4h (20-bar high/low)
    # Using rolling window with min_periods to avoid look-ahead
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume EMA for confirmation
    volume_ema_20_1d = pd.Series(volume_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ema_20_1d)
    
    # Calculate 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for trailing stop
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(abs(high - np.roll(close, 1)))
    tr3 = pd.Series(abs(low - np.roll(close, 1)))
    tr2.iloc[0] = tr2.iloc[1] if len(tr2) > 1 else 0
    tr3.iloc[0] = tr3.iloc[1] if len(tr3) > 1 else 0
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, min_periods=14, adjust=False).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ema_20_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume for confirmation (aligned to 4h)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm_1d = vol_1d_current[i] > 1.5 * volume_ema_20_1d_aligned[i]
        
        # Entry conditions
        long_entry = (close[i] > donchian_high[i] and 
                     vol_confirm_1d and 
                     close[i] > ema_50_1w_aligned[i])
        short_entry = (close[i] < donchian_low[i] and 
                      vol_confirm_1d and 
                      close[i] < ema_50_1w_aligned[i])
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
            elif short_entry:
                position = -1
                signals[i] = -0.25
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Update highest/lowest since entry
            if position == 1:
                highest_since_entry = max(highest_since_entry, close[i])
                lowest_since_entry = min(lowest_since_entry, close[i])
                # ATR trailing stop: exit if price drops 3*ATR from high
                if (close[i] < highest_since_entry - 3.0 * atr[i] or  # trailing stop
                    close[i] < donchian_low[i]):                     # opposite breakout
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                highest_since_entry = max(highest_since_entry, close[i])
                lowest_since_entry = min(lowest_since_entry, close[i])
                # ATR trailing stop: exit if price rises 3*ATR from low
                if (close[i] > lowest_since_entry + 3.0 * atr[i] or  # trailing stop
                    close[i] > donchian_high[i]):                    # opposite breakout
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals