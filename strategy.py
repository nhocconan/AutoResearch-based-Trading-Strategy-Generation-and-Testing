#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d HTF for direction (Camarilla H3/L3 breakout with volume confirmation) and 1h for entry timing
# In trending markets (ADX > 25): breakout trades in direction of trend
# In ranging markets (ADX < 20): mean reversion at Camarilla levels
# Uses discrete sizing 0.20 to limit trades to 15-37/year and control fee drag
# Session filter (08-20 UTC) reduces noise trades
# Works in bull/bear: breakout catches trends, mean reversion works in ranges, ADX regime filter avoids whipsaws

name = "1h_4h_1d_camarilla_breakout_adx_regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h ATR(14) for ADX
    tr1_4h = np.abs(high_4h[1:] - low_4h[:-1])
    tr2_4h = np.abs(high_4h[1:] - close_4h[:-1])
    tr3_4h = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_4h = wilders_smoothing(tr_4h, 14)
    
    # Calculate 4h +DM and -DM for ADX
    up_move_4h = high_4h[1:] - high_4h[:-1]
    down_move_4h = low_4h[:-1] - low_4h[1:]
    plus_dm_4h = np.where((up_move_4h > down_move_4h) & (up_move_4h > 0), up_move_4h, 0.0)
    minus_dm_4h = np.where((down_move_4h > up_move_4h) & (down_move_4h > 0), down_move_4h, 0.0)
    plus_dm_4h = np.concatenate([[np.nan], plus_dm_4h])
    minus_dm_4h = np.concatenate([[np.nan], minus_dm_4h])
    
    # Smoothed +DM, -DM, TR
    plus_dm_smooth_4h = wilders_smoothing(plus_dm_4h, 14)
    minus_dm_smooth_4h = wilders_smoothing(minus_dm_4h, 14)
    tr_smooth_4h = wilders_smoothing(tr_4h, 14)
    
    # Calculate 4h ADX(14)
    plus_di_4h = np.where(tr_smooth_4h != 0, 100 * plus_dm_smooth_4h / tr_smooth_4h, 0)
    minus_di_4h = np.where(tr_smooth_4h != 0, 100 * minus_dm_smooth_4h / tr_smooth_4h, 0)
    dx_4h = np.where((plus_di_4h + minus_di_4h) != 0, 
                     100 * np.abs(plus_di_4h - minus_di_4h) / (plus_di_4h + minus_di_4h), 
                     0)
    adx_4h = wilders_smoothing(dx_4h, 14)
    
    # Align 4h ADX to 1h
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Calculate 1d Camarilla pivot levels (based on prior day to avoid look-ahead)
    range_1d = high_1d - low_1d
    h3_1d = close_1d + 1.1 * range_1d
    l3_1d = close_1d - 1.1 * range_1d
    h4_1d = close_1d + 1.5 * range_1d
    l4_1d = close_1d - 1.5 * range_1d
    
    # Align 1d Camarilla levels to 1h
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Calculate 1h average volume (20-period)
    volume_s = pd.Series(volume)
    avg_volume_1h = volume_s.rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * avg_volume_1h
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or np.isnan(volume_confirmed[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Regime filter based on 4h ADX
        trending_regime = adx_4h_aligned[i] > 25
        ranging_regime = adx_4h_aligned[i] < 20
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if price breaks below L3 or we enter ranging regime
                if close[i] < l3_1d_aligned[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            elif ranging_regime:
                # Exit long if price rises above H3
                if close[i] > h3_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if price breaks above H3 or we enter ranging regime
                if close[i] > h3_1d_aligned[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
            elif ranging_regime:
                # Exit short if price drops below L3
                if close[i] < l3_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
        else:  # Flat
            if trending_regime:
                # Enter long on breakout above H3 with volume confirmation
                if close[i] > h3_1d_aligned[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.20
                # Enter short on breakout below L3 with volume confirmation
                elif close[i] < l3_1d_aligned[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.20
            elif ranging_regime:
                # Mean reversion: buy near L3, sell near H3
                if close[i] <= l3_1d_aligned[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.20
                elif close[i] >= h3_1d_aligned[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals