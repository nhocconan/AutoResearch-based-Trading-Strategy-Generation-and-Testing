#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13) measures buying/selling pressure
# 1d ADX > 25 indicates trending market, < 20 indicates ranging market
# In trending markets (ADX>25): Go long when Bull Power > 0 and rising, short when Bear Power < 0 and falling
# In ranging markets (ADX<20): Fade extremes - long when Bear Power < -std and turning up, short when Bull Power > std and turning down
# Volume confirmation ensures institutional participation
# Designed for low trade frequency (12-37/year) to minimize fee drag. Works in both bull and bear markets via regime adaptation.

name = "6h_ElderRay_1dADX_Regime_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter and EMA(13) for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate EMA(13) on 1d for Elder Ray
    ema_13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate ADX(14) on 1d for regime filter
    # ADX requires +DI, -DI, and DX calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period_adx = 14
    if len(plus_dm) >= period_adx:
        plus_di_smoothed = wilder_smooth(plus_dm, period_adx)
        minus_di_smoothed = wilder_smooth(minus_dm, period_adx)
        tr_smoothed = wilder_smooth(tr, period_adx)
        
        # Avoid division by zero
        plus_di = np.where(tr_smoothed != 0, plus_di_smoothed / tr_smoothed * 100, 0)
        minus_di = np.where(tr_smoothed != 0, minus_di_smoothed / tr_smoothed * 100, 0)
        
        dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
        adx = wilder_smooth(dx[period_adx-1:], period_adx) if len(dx) >= period_adx else np.array([])
        
        # Pad arrays to match original length
        plus_di_padded = np.concatenate([np.full(period_adx, np.nan), plus_di])
        minus_di_padded = np.concatenate([np.full(period_adx, np.nan), minus_di])
        adx_padded = np.concatenate([np.full(2*period_adx-1, np.nan), adx])
    else:
        plus_di_padded = np.full(len(high_1d), np.nan)
        minus_di_padded = np.full(len(high_1d), np.nan)
        adx_padded = np.full(len(high_1d), np.nan)
    
    # Align 1d indicators to 6h timeframe (wait for completed 1d bar)
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_padded)
    
    # Calculate Elder Ray on 6h using aligned 1d EMA(13)
    bull_power = high - ema_13_1d_aligned  # High - EMA(13)
    bear_power = low - ema_13_1d_aligned   # Low - EMA(13)
    
    # Volume confirmation (1.5x 20-period average) on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 50  # max(30 for 1d EMA/ADX, 20 for volume MA +1 for shift)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_1d_aligned[i]
        bp = bear_power[i]
        bl = bull_power[i]
        
        if position == 0:  # Flat - look for new entries
            # Trending market (ADX > 25): trade with momentum
            if adx_val > 25:
                # Long: Bull Power > 0 and rising (momentum building)
                if i > start_idx and bl > 0 and bl > bull_power[i-1] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power < 0 and falling (momentum building)
                elif i > start_idx and bp < 0 and bp < bear_power[i-1] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            # Ranging market (ADX < 20): fade extremes
            elif adx_val < 20:
                # Calculate standard deviation of Bear Power for dynamic threshold
                lookback = min(50, i)
                if lookback >= 10:
                    bp_std = np.nanstd(bear_power[i-lookback:i])
                    bl_std = np.nanstd(bull_power[i-lookback:i])
                    
                    # Long: Bear Power < -1*std and turning up (oversold bounce)
                    if bp < -bp_std and i > start_idx and bp > bear_power[i-1] and volume_spike[i]:
                        signals[i] = 0.25
                        position = 1
                    # Short: Bull Power > 1*std and turning down (overbought fade)
                    elif bl > bl_std and i > start_idx and bl < bull_power[i-1] and volume_spike[i]:
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_signal = False
            # Trending market: exit when Bull Power turns negative (momentum lost)
            if adx_val > 25 and bl <= 0:
                exit_signal = True
            # Ranging market: exit when Bear Power > 0 (mean reversion complete)
            elif adx_val < 20 and bp >= 0:
                exit_signal = True
            # Universal exit: volume spike in opposite direction
            elif volume_spike[i] and bl < 0:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_signal = False
            # Trending market: exit when Bear Power turns positive (momentum lost)
            if adx_val > 25 and bp >= 0:
                exit_signal = True
            # Ranging market: exit when Bull Power < 0 (mean reversion complete)
            elif adx_val < 20 and bl <= 0:
                exit_signal = True
            # Universal exit: volume spike in opposite direction
            elif volume_spike[i] and bp > 0:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals