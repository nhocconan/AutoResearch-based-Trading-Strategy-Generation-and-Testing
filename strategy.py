#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d ADX trend filter and volume confirmation.
# Long when Williams %R crosses above -80 from below (oversold bounce) AND 1d ADX > 25 (trending market) AND volume > 1.5x 20-bar average.
# Short when Williams %R crosses below -20 from above (overbought rejection) AND 1d ADX > 25 AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 6h timeframe to avoid overtrading.
# Works in bull (buy oversold bounces in uptrend) and bear (sell overbought rejections in downtrend) via ADX trend filter.

name = "6h_WilliamsR_Reversal_1dADX_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ADX(14) for trend filter
    # Calculate True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['low'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean()
    
    # Calculate +DM and -DM
    up_move = pd.Series(df_1d['high']).diff()
    down_move = pd.Series(df_1d['low']).diff().abs()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth +DM, -DM, and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean()
    atr_smooth = atr.ewm(span=14, adjust=False, min_periods=14).mean()
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm_smooth / atr_smooth)
    minus_di = 100 * (minus_dm_smooth / atr_smooth)
    
    # Calculate DX and ADX
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.ewm(span=14, adjust=False, min_periods=14).mean()
    adx_values = adx.values
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for Williams %R and ADX calculation
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 6h timeframe
        hour = hours[i]
        
        if np.isnan(adx_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_adx_1d = adx_1d_aligned[i]
        
        # Calculate Williams %R(14)
        if i < 14 + start_idx:
            signals[i] = 0.0
            continue
            
        highest_high = np.max(high[i-13:i+1])  # 14-period high including current
        lowest_low = np.min(low[i-13:i+1])    # 14-period low including current
        
        if highest_high == lowest_low:
            williams_r = -50  # avoid division by zero
        else:
            williams_r = -100 * (highest_high - curr_close) / (highest_high - lowest_low)
        
        # Calculate Williams %R previous value for crossover detection
        if i-1 < 14 + start_idx:
            prev_williams_r = -50
        else:
            prev_highest_high = np.max(high[i-14:i])  # 14-period high excluding current
            prev_lowest_low = np.min(low[i-14:i])     # 14-period low excluding current
            if prev_highest_high == prev_lowest_low:
                prev_williams_r = -50
            else:
                prev_williams_r = -100 * (prev_highest_high - close[i-1]) / (prev_highest_high - prev_lowest_low)
        
        # Volume confirmation: current 6h volume > 1.5x 20-bar average
        if i < 20 + start_idx:
            signals[i] = 0.0
            continue
            
        vol_ma = np.mean(volume[i-20:i])  # 20-period simple moving average
        if vol_ma <= 0:
            signals[i] = 0.0
            continue
        volume_confirm = curr_vol > (vol_ma * 1.5)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 from below AND ADX > 25 AND volume confirmation
            if (prev_williams_r <= -80 and williams_r > -80 and 
                curr_adx_1d > 25 and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above AND ADX > 25 AND volume confirmation
            elif (prev_williams_r >= -20 and williams_r < -20 and 
                  curr_adx_1d > 25 and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses above -20 (overbought) OR ADX < 20 (trend weakening)
            if (williams_r >= -20 or 
                curr_adx_1d < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -80 (oversold) OR ADX < 20 (trend weakening)
            if (williams_r <= -80 or 
                curr_adx_1d < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals