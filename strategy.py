#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R momentum with 12h volume confirmation and 1d ADX trend filter.
# Long when Williams %R crosses above -80 (oversold) AND volume > 1.3x 12h average AND ADX > 25 (trending market)
# Short when Williams %R crosses below -20 (overbought) AND volume > 1.3x 12h average AND ADX > 25
# Exit when Williams %R crosses -50 (mean reversion) or ADX < 20 (trend ends)
# Uses momentum for entry timing, volume for confirmation, ADX for trend strength filtering.
# Target: 20-30 trades/year per symbol.
name = "4h_WilliamsR_Volume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for volume average
    df_12h = get_htf_data(prices, '12h')
    vol_ma_12h = pd.Series(df_12h['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    # Calculate True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Calculate +DM and -DM
    up_move = df_1d['high'].diff()
    down_move = df_1d['low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    # Directional indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or np.isnan(vol_ma_12h_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ma = vol_ma_12h_aligned[i]
        adx_val = adx_aligned[i]
        wr = williams_r[i]
        vol = volume[i]
        
        # Trend filter: only trade when ADX > 25 (strong trend)
        strong_trend = adx_val > 25
        
        if position == 0:
            # Long entry: Williams %R crosses above -80 from below + volume spike + strong trend
            if i > start_idx and williams_r[i-1] <= -80 and wr > -80 and vol > 1.3 * vol_ma and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R crosses below -20 from above + volume spike + strong trend
            elif i > start_idx and williams_r[i-1] >= -20 and wr < -20 and vol > 1.3 * vol_ma and strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses below -50 OR ADX < 20 (trend weakening)
            if wr < -50 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses above -50 OR ADX < 20 (trend weakening)
            if wr > -50 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals