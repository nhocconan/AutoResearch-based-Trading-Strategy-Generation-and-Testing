#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume confirmation and ATR-based stoploss
# Donchian breakouts capture strong momentum moves in both bull and bear markets
# Volume confirmation (>1.5x 20-period average) ensures institutional participation
# ATR stoploss (2.5x ATR) limits drawdown during reversals
# Primary timeframe 12h reduces trade frequency to avoid fee drag (target: 50-150 trades over 4 years)
# 1d EMA50 trend filter ensures alignment with higher timeframe direction

name = "12h_Donchian20_VolumeSpike_ATR_Stop_1dEMA50_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20)
    low_roll = pd.Series(low).rolling(window=20, min_periods=20)
    donchian_high = high_roll.max().values
    donchian_low = low_roll.min().values
    
    # Calculate ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = np.abs(high[0] - close[0])  # First period: use high-low
    tr3.iloc[0] = np.abs(low[0] - close[0])
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = max(20, 14, 20, 50)  # warmup for Donchian, ATR, volume MA, 1d EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50 = ema_50_aligned[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish breakout: price breaks above Donchian high with price above 1d EMA50
                if curr_high > donchian_high[i-1] and curr_close > curr_ema_50:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    atr_stop = entry_price - 2.5 * curr_atr
                # Bearish breakout: price breaks below Donchian low with price below 1d EMA50
                elif curr_low < donchian_low[i-1] and curr_close < curr_ema_50:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    atr_stop = entry_price + 2.5 * curr_atr
        
        elif position == 1:  # Long position
            # Exit on Donchian low break or ATR stoploss
            if curr_low < donchian_low[i-1] or curr_close < atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Trail stoploss: raise stop if price moves favorably
                atr_stop = max(atr_stop, curr_close - 2.5 * curr_atr)
        
        elif position == -1:  # Short position
            # Exit on Donchian high break or ATR stoploss
            if curr_high > donchian_high[i-1] or curr_close > atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Trail stoploss: lower stop if price moves favorably
                atr_stop = min(atr_stop, curr_close + 2.5 * curr_atr)
    
    return signals