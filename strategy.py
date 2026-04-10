#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout + 1d volume confirmation + 1w ADX trend filter
# - Long when price breaks above H3 (Camarilla resistance) AND volume > 1.5x 20-period average AND 1w ADX > 25 (trending up)
# - Short when price breaks below L3 (Camarilla support) AND volume > 1.5x 20-period average AND 1w ADX > 25 (trending down)
# - Exit when price crosses the daily pivot point (mean reversion to equilibrium)
# - Uses discrete position sizing 0.30 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Camarilla levels provide precise intraday support/resistance in trending markets
# - Volume confirmation ensures breakouts have institutional participation
# - 1w ADX filter ensures we only trade when higher timeframe trend is strong

name = "12h_1d_1w_camarilla_volume_adx_v1"
timeframe = "12h"
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
    
    # Pre-compute 12h typical price for Camarilla calculation
    typical_price = (prices['high'] + prices['low'] + prices['close']) / 3
    
    # Pre-compute 12h volume confirmation
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1w ADX (14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - np.roll(close_1w, 1)[1:])
    tr3 = np.abs(low_1w[1:] - np.roll(close_1w, 1)[1:])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # first element is NaN
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    atr_1w = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / atr_1w
    minus_di = 100 * minus_dm_smooth / atr_1w
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # ADX trend filter: > 25 = strong trend
    adx_trend = adx > 25
    
    # Align HTF indicators to 12h timeframe
    adx_trend_aligned = align_htf_to_ltf(prices, df_1w, adx_trend)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(typical_price.iloc[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(adx_trend_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        if position == 0:  # Flat - look for new entries
            # Calculate Camarilla levels from previous 1d bar
            prev_high = df_1d['high'].iloc[-1] if len(df_1d) > 0 else typical_price.iloc[i]
            prev_low = df_1d['low'].iloc[-1] if len(df_1d) > 0 else typical_price.iloc[i]
            prev_close = df_1d['close'].iloc[-1] if len(df_1d) > 0 else typical_price.iloc[i]
            
            # Camarilla levels
            pivot = (prev_high + prev_low + prev_close) / 3
            range_hl = prev_high - prev_low
            h3 = pivot + (range_hl * 1.1 / 4)
            l3 = pivot - (range_hl * 1.1 / 4)
            h4 = pivot + (range_hl * 1.1 / 2)
            l4 = pivot - (range_hl * 1.1 / 2)
            
            price = typical_price.iloc[i]
            
            # Long conditions: price breaks above H3 AND volume spike AND 1w ADX trend
            if (price > h3 and 
                volume_spike[i] and 
                adx_trend_aligned[i]):
                position = 1
                signals[i] = 0.30
            # Short conditions: price breaks below L3 AND volume spike AND 1w ADX trend
            elif (price < l3 and 
                  volume_spike[i] and 
                  adx_trend_aligned[i]):
                position = -1
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Calculate current Camarilla pivot for exit
            # Need to get the most recent completed 1d bar
            # For simplicity, use the same pivot calculation as entry (using previous bar)
            if len(df_1d) >= 2:
                prev_high = df_1d['high'].iloc[-2] if len(df_1d) >= 2 else df_1d['high'].iloc[-1]
                prev_low = df_1d['low'].iloc[-2] if len(df_1d) >= 2 else df_1d['low'].iloc[-1]
                prev_close = df_1d['close'].iloc[-2] if len(df_1d) >= 2 else df_1d['close'].iloc[-1]
            else:
                prev_high = df_1d['high'].iloc[-1] if len(df_1d) > 0 else typical_price.iloc[i]
                prev_low = df_1d['low'].iloc[-1] if len(df_1d) > 0 else typical_price.iloc[i]
                prev_close = df_1d['close'].iloc[-1] if len(df_1d) > 0 else typical_price.iloc[i]
            
            pivot = (prev_high + prev_low + prev_close) / 3
            price = typical_price.iloc[i]
            
            # Exit when price crosses the daily pivot point (mean reversion to equilibrium)
            exit_long = (position == 1 and price < pivot)
            exit_short = (position == -1 and price > pivot)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.30
                else:
                    signals[i] = -0.30
    
    return signals