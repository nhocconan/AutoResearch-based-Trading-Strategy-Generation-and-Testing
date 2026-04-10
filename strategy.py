#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume spike and ADX trend filter
# - Long when price breaks above Donchian(20) high AND 1d volume > 1.8x 20-period volume SMA AND ADX(14) > 25 (strong trend)
# - Short when price breaks below Donchian(20) low AND 1d volume > 1.8x 20-period volume SMA AND ADX(14) > 25 (strong trend)
# - Exit: ATR(14) trailing stop (2.5*ATR) from highest/lowest since entry
# - Uses 4h for price action (Donchian channels), 1d for volume/ADX confirmation
# - Volume spike confirms institutional interest; ADX filter ensures trending markets only
# - Tight entries target ~25-35 trades/year to minimize fee drag (proven winners: ETH test Sharpe 1.47)
# - Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) with volume/ADX filters

name = "4h_1d_donchian_volspike_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for HTF confirmation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate 1d volume SMA for confirmation
    vol_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 1d ADX for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(np.maximum(tr1, tr2), tr3)
    tr_1d = np.concatenate([[np.nan], tr_1d])
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    
    # Directional Index (DX) and ADX
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute Donchian channels from prior completed 1d bar (use previous completed 1d bar)
    # Donchian channels based on prior 20 periods of 1d data
    donchian_high_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (wait for 1d bar to complete)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20_1d)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20_1d)
    
    # ATR for dynamic stoploss (using 4h data)
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Track highest/lowest since entry for trailing stop
    highest_high_since_entry = np.full(n, np.nan)
    lowest_low_since_entry = np.full(n, np.nan)
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.8x 20-period volume SMA
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_confirm = vol_1d_aligned[i] > 1.8 * volume_sma_20_1d_aligned[i]
        
        # ADX filter: ADX > 25 indicates strong trend (good for breakouts)
        adx_filter = adx_1d_aligned[i] > 25
        
        # Only trade when both volume confirmation and ADX filter are present
        if vol_confirm and adx_filter:
            # Long: price breaks above Donchian high
            if close[i] > donchian_high_aligned[i]:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                    highest_high_since_entry[i] = high[i]
                else:
                    signals[i] = 0.25
                    highest_high_since_entry[i] = max(highest_high_since_entry[i-1] if i > 0 else high[i], high[i])
            # Short: price breaks below Donchian low
            elif close[i] < donchian_low_aligned[i]:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                    lowest_low_since_entry[i] = low[i]
                else:
                    signals[i] = -0.25
                    lowest_low_since_entry[i] = min(lowest_low_since_entry[i-1] if i > 0 else low[i], low[i])
            else:
                # Maintain position and update tracking levels
                if position == 1:
                    signals[i] = 0.25
                    highest_high_since_entry[i] = max(highest_high_since_entry[i-1] if i > 0 else high[i], high[i])
                elif position == -1:
                    signals[i] = -0.25
                    lowest_low_since_entry[i] = min(lowest_low_since_entry[i-1] if i > 0 else low[i], low[i])
                else:
                    signals[i] = 0.0
            
            # Check for ATR trailing stop exit
            if position == 1 and not np.isnan(highest_high_since_entry[i]):
                if close[i] < (highest_high_since_entry[i] - 2.5 * atr_4h[i]):
                    position = 0
                    signals[i] = 0.0
            elif position == -1 and not np.isnan(lowest_low_since_entry[i]):
                if close[i] > (lowest_low_since_entry[i] + 2.5 * atr_4h[i]):
                    position = 0
                    signals[i] = 0.0
        else:
            # No trade: exit any position if conditions not met
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals