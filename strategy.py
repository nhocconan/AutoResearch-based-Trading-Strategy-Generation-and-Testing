#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power + 1d Regime Filter
# Uses 1d ADX to identify trending vs ranging markets, then applies Elder Ray (Bull/Bear Power) for entries.
# In strong trends (ADX > 25): go long when Bear Power turns positive (bulls overcoming bears), short when Bull Power turns negative.
# In ranging markets (ADX < 20): fade extremes - long when Bull Power crosses above -1.5*ATR, short when Bear Power crosses below 1.5*ATR.
# Volume confirmation reduces false signals. Target: 50-150 trades over 4 years.
# Works in both bull and bear markets by adapting to regime.

name = "6h_ElderRay_Power_1dADX_Regime_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d indicators for regime and Elder Ray components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA13 for Elder Ray calculation
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 1d ATR for Elder Ray normalization and regime ADX
    def calculate_atr(high, low, close, period=14):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # 1d ADX for regime detection (trending vs ranging)
    def calculate_adx(high, low, close, period=14):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Wilder's smoothing
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] - (result[i-1]/period) + data[i]
            return result
        
        atr = wilder_smooth(tr, period)
        plus_di = 100 * wilder_smooth(plus_dm, period) / atr
        minus_di = 100 * wilder_smooth(minus_dm, period) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilder_smooth(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Elder Ray Components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align all 1d indicators to 6h timeframe
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: current 6h volume > 1.8x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema13_1d_aligned[i]) or np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or \
           np.isnan(atr_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.8)  # Volume spike threshold
        
        # Regime detection
        is_trending = adx_1d_aligned[i] > 25
        is_ranging = adx_1d_aligned[i] < 20
        
        # Elder Ray signals
        bull_power = bull_power_1d_aligned[i]
        bear_power = bear_power_1d_aligned[i]
        atr_val = atr_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if is_trending:
                # In trending markets: follow Elder Ray momentum
                # Long when Bear Power turns positive (bulls overcoming bears)
                # Short when Bull Power turns negative (bears overcoming bulls)
                if bear_power > 0 and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                elif bull_power < 0 and volume_confirm:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif is_ranging:
                # In ranging markets: fade extremes
                # Long when Bull Power crosses above -1.5*ATR (oversold)
                # Short when Bear Power crosses below 1.5*ATR (overbought)
                if bull_power > (-1.5 * atr_val) and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                elif bear_power < (1.5 * atr_val) and volume_confirm:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Transition regime - no trades
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions
            if is_trending:
                # Exit long when Bull Power turns negative (trend weakening)
                if bull_power < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif is_ranging:
                # Exit long when Bull Power crosses below zero (mean reversion complete)
                if bull_power < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # Transition regime - exit
                signals[i] = 0.0
                position = 0
        
        elif position == -1:  # Short position
            # Exit conditions
            if is_trending:
                # Exit short when Bear Power turns positive (trend weakening)
                if bear_power > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif is_ranging:
                # Exit short when Bear Power crosses above zero (mean reversion complete)
                if bear_power > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Transition regime - exit
                signals[i] = 0.0
                position = 0
    
    return signals