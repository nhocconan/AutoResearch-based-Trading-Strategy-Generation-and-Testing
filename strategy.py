#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band breakout with 1d volume spike and 1w ADX regime filter
# - Long when close breaks above upper BB(20,2) with volume > 1.3x 20-day EMA and 1w ADX > 25
# - Short when close breaks below lower BB(20,2) with volume > 1.3x 20-day EMA and 1w ADX > 25
# - Exit: ATR trailing stop (2.5x ATR) or price reverts to middle BB
# - Position sizing: 0.30 discrete level
# - Targets ~25-35 trades/year on 4h timeframe. BB breakouts capture volatility expansion,
#   volume confirmation avoids fakeouts, 1w ADX > 25 ensures strong trending environment.
#   Works in bull/bear: breakouts work in both directions with trend filter improving win rate.

name = "4h_1d_1w_bb_breakout_volume_adx_v1"
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
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Calculate Bollinger Bands(20,2) on 4h
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Calculate 1d volume EMA for confirmation (20-period)
    volume_ema_20_1d = pd.Series(volume_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ema_20_1d)
    
    # Calculate 1w ADX(14) for trend filter
    high_1w_series = pd.Series(high_1w)
    low_1w_series = pd.Series(low_1w)
    close_1w_series = pd.Series(close_1w)
    
    up_move = high_1w_series.diff()
    down_move = low_1w_series.diff().mul(-1)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr2[1] if len(tr2) > 1 else 0
    tr3[0] = tr3[1] if len(tr3) > 1 else 0
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_1w = pd.Series(tr_1w).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr_1w
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr_1w
    
    dx = np.abs(plus_di - minus_di) / (np.abs(plus_di) + np.abs(minus_di)) * 100
    dx = np.where(np.isnan(dx), 0, dx)
    adx_1w = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate ATR(14) for trailing stop on 4h
    tr1_4h = high - low
    tr2_4h = np.abs(high - np.roll(close, 1))
    tr3_4h = np.abs(low - np.roll(close, 1))
    tr2_4h[0] = tr2_4h[1] if len(tr2_4h) > 1 else 0
    tr3_4h[0] = tr3_4h[1] if len(tr3_4h) > 1 else 0
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_4h = pd.Series(tr_4h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_middle[i]) or
            np.isnan(volume_ema_20_1d_aligned[i]) or np.isnan(adx_1w_aligned[i]) or
            np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume for confirmation (aligned to 4h)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = vol_1d_current[i] > 1.3 * volume_ema_20_1d_aligned[i]
        
        # Regime filter: 1w ADX > 25 indicates strong trending market
        regime_filter = adx_1w_aligned[i] > 25
        
        # Entry conditions
        long_entry = (close[i] > bb_upper[i] and  # Break above upper BB
                     vol_confirm and 
                     regime_filter)
        short_entry = (close[i] < bb_lower[i] and  # Break below lower BB
                      vol_confirm and 
                      regime_filter)
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.30
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
            elif short_entry:
                position = -1
                signals[i] = -0.30
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Update highest/lowest since entry
            if position == 1:
                highest_since_entry = max(highest_since_entry, close[i])
                lowest_since_entry = min(lowest_since_entry, close[i])
                # ATR trailing stop: exit if price drops 2.5*ATR from high
                # or price reverts to middle BB (mean reversion)
                if (close[i] < highest_since_entry - 2.5 * atr_4h[i] or  # trailing stop
                    close[i] <= bb_middle[i]):         # BB mean reversion exit
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.30
            else:  # position == -1 (Short position)
                highest_since_entry = max(highest_since_entry, close[i])
                lowest_since_entry = min(lowest_since_entry, close[i])
                # ATR trailing stop: exit if price rises 2.5*ATR from low
                # or price reverts to middle BB (mean reversion)
                if (close[i] > lowest_since_entry + 2.5 * atr_4h[i] or  # trailing stop
                    close[i] >= bb_middle[i]):         # BB mean reversion exit
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.30
    
    return signals