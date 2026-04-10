#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d volume confirmation and ADX trend filter
# - Long when price breaks above 20-period Donchian high + 1d volume > 1.5x 20-period volume SMA + ADX > 25 (strong trend)
# - Short when price breaks below 20-period Donchian low + 1d volume > 1.5x 20-period volume SMA + ADX > 25 (strong trend)
# - Exit: price returns to opposite Donchian level or ADX < 20 (weak trend/ranging)
# - Position sizing: 0.25 discrete level
# - Works in bull/bear: Donchian adapts to volatility, volume confirms institutional interest, ADX filter avoids whipsaws in ranging markets
# - 12h timeframe balances signal frequency with cost efficiency (target: 12-37 trades/year)

name = "12h_1d_donchian_volume_adx_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 20-period Donchian channels on 12h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume SMA(20) for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate ADX(14) on 12h for trend filter
    # True Range
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]  # first bar
    # Plus Directional Movement
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    plus_dm[0] = 0
    # Minus Directional Movement
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    minus_dm[0] = 0
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr = pd.Series(tr1).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period SMA
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = vol_1d_current[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend (good for breakouts)
        strong_trend = adx[i] > 25
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # Exit conditions: return to opposite Donchian level or ADX < 20 (weak trend)
        long_exit = close[i] < donchian_low[i] or adx[i] < 20
        short_exit = close[i] > donchian_high[i] or adx[i] < 20
        
        # Entry conditions
        long_entry = long_breakout and vol_confirm and strong_trend
        short_entry = short_breakout and vol_confirm and strong_trend
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals