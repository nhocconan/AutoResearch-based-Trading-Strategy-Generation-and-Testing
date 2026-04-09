#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla pivot levels with volume confirmation and chop regime filter
# Camarilla pivots provide structured support/resistance levels based on previous 4h bar's range
# Long when price breaks above H3 with volume confirmation in low chop (trending) regime
# Short when price breaks below L3 with volume confirmation in low chop regime
# In high chop (ranging) regime, fade extremes: long at L3, short at H3
# Uses 4h/1d for signal direction, 1h only for entry timing to reduce trade frequency
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe
# Session filter: 08-20 UTC to avoid low-liquidity periods
# Discrete position sizing 0.20 to minimize fee churn

name = "1h_4h_1d_camarilla_breakout_v1"
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
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla pivot levels
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low), 
    #            L3 = close - 1.1*(high-low), L4 = close - 1.5*(high-low)
    range_4h = high_4h - low_4h
    camarilla_h3 = close_4h + 1.1 * range_4h
    camarilla_l3 = close_4h - 1.1 * range_4h
    camarilla_h4 = close_4h + 1.5 * range_4h
    camarilla_l4 = close_4h - 1.5 * range_4h
    
    # Calculate 4h ATR(14) for volatility filter
    tr1 = np.abs(high_4h[1:] - low_4h[:-1])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
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
    
    atr_4h = wilders_smoothing(tr, 14)
    atr_10_4h = wilders_smoothing(tr, 10)  # For volatility filter
    
    # Calculate 4h average volume (20-period)
    if 'volume' in df_4h.columns:
        volume_4h = df_4h['volume'].values
    else:
        volume_4h = np.zeros_like(close_4h)  # fallback
    
    vol_s_4h = pd.Series(volume_4h)
    avg_vol_4h = vol_s_4h.rolling(window=20, min_periods=20).mean().values
    
    # Calculate Bollinger Band Width for chop regime filter (using 4h data)
    close_s_4h = pd.Series(close_4h)
    basis_4h = close_s_4h.rolling(window=20, min_periods=20).mean().values
    dev_4h = close_s_4h.rolling(window=20, min_periods=20).std().values
    upper_bb_4h = basis_4h + 2.0 * dev_4h
    lower_bb_4h = basis_4h - 2.0 * dev_4h
    bb_width_4h = (upper_bb_4h - lower_bb_4h) / basis_4h
    bb_width_4h = np.where(basis_4h != 0, bb_width_4h, 0)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Calculate 1d EMA(200) for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align all indicators to 1h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4)
    avg_vol_4h_aligned = align_htf_to_ltf(prices, df_4h, avg_vol_4h)
    bb_width_4h_aligned = align_htf_to_ltf(prices, df_4h, bb_width_4h)
    atr_10_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_10_4h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(avg_vol_4h_aligned[i]) or np.isnan(bb_width_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume (using 1h volume)
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # Chop regime: low BB width = trending, high BB width = ranging
        # Using 4h BB width aligned to 1h
        trending_regime = bb_width_4h_aligned[i] < 0.05  # Low volatility = trending
        ranging_regime = bb_width_4h_aligned[i] > 0.10   # High volatility = ranging
        
        # Trend filter: EMA50 > EMA200 on 1d = uptrend, EMA50 < EMA200 = downtrend
        uptrend = ema_50_1d_aligned[i] > ema_200_1d_aligned[i]
        downtrend = ema_50_1d_aligned[i] < ema_200_1d_aligned[i]
        
        if position == 1:  # Long position
            if trending_regime and volume_confirmed and uptrend:
                # Exit long if price falls below H3
                if close[i] < camarilla_h3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            elif ranging_regime:
                # Exit long if price moves back above L3 (mean reversion exit)
                if close[i] > camarilla_l3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
                
        elif position == -1:  # Short position
            if trending_regime and volume_confirmed and downtrend:
                # Exit short if price rises above L3
                if close[i] > camarilla_l3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
            elif ranging_regime:
                # Exit short if price moves back below H3 (mean reversion exit)
                if close[i] < camarilla_h3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
        else:  # Flat
            if trending_regime and volume_confirmed:
                # Breakout strategy in trending market (only follow higher timeframe trend)
                if uptrend and close[i] > camarilla_h3_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                elif downtrend and close[i] < camarilla_l3_aligned[i]:
                    position = -1
                    signals[i] = -0.20
            elif ranging_regime:
                # Mean reversion at extremes in ranging market
                if close[i] < camarilla_l3_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                elif close[i] > camarilla_h3_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals