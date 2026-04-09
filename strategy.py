#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation
# In trending regimes (ADX > 25): trade in direction of 1d Elder Ray with 6h pullback to EMA21
# In ranging regimes (ADX < 20): fade extreme 6h Elder Ray readings with volume confirmation
# Uses discrete position sizing 0.25 to limit trades to ~12-37/year and reduce fee drag
# Elder Ray captures institutional buying/selling pressure; ADX filters whipsaws in ranging markets

name = "6h_1d_elder_ray_volume_adx_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR(14) for ADX
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # Calculate 1d +DM and -DM for ADX
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed +DM, -DM, and TR
    atr_period = 14
    smoothed_tr = wilders_smoothing(tr, atr_period)
    smoothed_plus_dm = wilders_smoothing(plus_dm, atr_period)
    smoothed_minus_dm = wilders_smoothing(minus_dm, atr_period)
    
    # Calculate +DI and -DI
    plus_di = 100 * smoothed_plus_dm / smoothed_tr
    minus_di = 100 * smoothed_minus_dm / smoothed_tr
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, atr_period)
    
    # Calculate 1d Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    close_s_1d = pd.Series(close_1d)
    ema13_1d = close_s_1d.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Calculate 6h EMA21 for pullback entries
    close_s = pd.Series(close)
    ema21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 6h Elder Ray for extreme readings
    ema13_6h = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_6h = high - ema13_6h
    bear_power_6h = low - ema13_6h
    
    # Calculate 6h average volume (20-period)
    volume_s = pd.Series(volume)
    avg_volume_6h = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i]) or np.isnan(ema21[i]) or
            np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or
            np.isnan(avg_volume_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_6h[i]
        
        # Regime filter
        trending_regime = adx_aligned[i] > 25
        ranging_regime = adx_aligned[i] < 20
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if bear power turns negative or price drops below EMA21
                if bear_power_1d_aligned[i] < 0 or close[i] < ema21[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # Exit long if bull power becomes extremely high (overbought)
                if bull_power_6h[i] > np.nanpercentile(bull_power_6h[max(0, i-50):i+1], 80):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if bull power turns positive or price rises above EMA21
                if bull_power_1d_aligned[i] > 0 or close[i] > ema21[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # Exit short if bear power becomes extremely low (oversold)
                if bear_power_6h[i] < np.nanpercentile(bear_power_6h[max(0, i-50):i+1], 20):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending_regime:
                # Enter long on bull power > 0 with pullback to EMA21 and volume confirmation
                if bull_power_1d_aligned[i] > 0 and close[i] <= ema21[i] * 1.005 and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                # Enter short on bear power < 0 with pullback to EMA21 and volume confirmation
                elif bear_power_1d_aligned[i] < 0 and close[i] >= ema21[i] * 0.995 and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean reversion: buy extreme bear power, sell extreme bull power
                bull_extreme = bull_power_6h[i] > np.nanpercentile(bull_power_6h[max(0, i-50):i+1], 80)
                bear_extreme = bear_power_6h[i] < np.nanpercentile(bear_power_6h[max(0, i-50):i+1], 20)
                
                if bear_extreme and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif bull_extreme and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
    
    return signals