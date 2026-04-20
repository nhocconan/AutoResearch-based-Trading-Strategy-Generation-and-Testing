#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly trend filter + daily breakout with volume confirmation
# Uses 1w EMA for trend direction, 1d Donchian breakout for entry, volume spike for confirmation
# Designed to work in both bull and bear markets by following higher timeframe trend
# Target: 15-25 trades/year, low frequency to minimize fee drag
name = "1d_WeeklyTrend_DonchianBreakout_VolumeConfirmation_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend
    df_1w = get_htf_data(prices, '1w')
    # Get daily data ONCE before loop for Donchian and volume
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 2 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly EMA34 for trend direction
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Daily average volume (20-period) for volume confirmation
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Daily ATR for exit (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        ema_trend = ema_34_1w_aligned[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        vol_avg = vol_avg_1d_aligned[i]
        current_atr = atr[i]
        current_close = prices['close'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_trend) or np.isnan(donch_high_val) or np.isnan(donch_low_val) or \
           np.isnan(vol_avg) or np.isnan(current_atr):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5x daily average volume
        vol_spike = current_volume > 1.5 * vol_avg
        
        if position == 0:
            # Long: price breaks above Donchian high with volume spike and weekly uptrend
            if current_close > donch_high_val and vol_spike and current_close > ema_trend:
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            # Short: price breaks below Donchian low with volume spike and weekly downtrend
            elif current_close < donch_low_val and vol_spike and current_close < ema_trend:
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or ATR stop loss
            if current_close < donch_low_val:
                signals[i] = 0.0
                position = 0
            elif current_close < entry_price - 2.5 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or ATR stop loss
            if current_close > donch_high_val:
                signals[i] = 0.0
                position = 0
            elif current_close > entry_price + 2.5 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals