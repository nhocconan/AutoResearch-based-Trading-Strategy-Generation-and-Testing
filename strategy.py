#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based position sizing
# Donchian breakouts capture strong momentum moves; 1d EMA50 filter ensures alignment with higher timeframe trend
# ATR-based sizing (0.25 at 1x ATR volatility) normalizes risk across different market regimes
# Designed for ~20-50 trades/year to minimize fee drag while maintaining edge in both bull and bear markets
# Works in bull/bear via 1d EMA50 trend filter - only trades in direction of higher timeframe momentum
# Uses volume confirmation (>1.5x 20-period average) to reduce false breakouts

name = "4h_Donchian20_1dEMA50_ATRSize_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR (20-period) for volatility-based sizing and stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Donchian/volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = curr_volume > 1.5 * curr_vol_ma
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit (2x ATR) or price breaks below Donchian low
            if curr_close < curr_close - 2.0 * curr_atr or curr_close < curr_donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                # ATR-based position sizing: 0.25 at 1x ATR volatility
                vol_factor = min(curr_atr / (curr_close * 0.02), 2.0)  # normalize to ~2% daily vol
                size = 0.25 * vol_factor
                signals[i] = np.clip(size, 0.15, 0.35)  # keep within reasonable bounds
                
        elif position == -1:  # Short position
            # Exit: stoploss hit (2x ATR) or price breaks above Donchian high
            if curr_close > curr_close + 2.0 * curr_atr or curr_close > curr_donchian_high:
                signals[i] = 0.0
                position = 0
            else:
                # ATR-based position sizing: 0.25 at 1x ATR volatility
                vol_factor = min(curr_atr / (curr_close * 0.02), 2.0)  # normalize to ~2% daily vol
                size = 0.25 * vol_factor
                signals[i] = -np.clip(size, 0.15, 0.35)  # keep within reasonable bounds
                
        else:  # Flat - look for new breakout entries
            # Long breakout when price closes above Donchian high with 1d EMA50 uptrend and volume confirmation
            if curr_close > curr_donchian_high and curr_close > curr_ema50_1d and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short breakout when price closes below Donchian low with 1d EMA50 downtrend and volume confirmation
            elif curr_close < curr_donchian_low and curr_close < curr_ema50_1d and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals