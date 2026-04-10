#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h/1d trend filter and session filter
# - Primary timeframe: 1h for precise entry timing
# - HTF: 4h for trend direction (EMA50), 1d for regime filter (ADX > 25 = trending)
# - Entry: Price breaks above/below Camarilla H3/L3 levels with volume confirmation
# - Volume: 1h volume > 1.5x 20-period average
# - Session: Trade only 08-20 UTC to avoid low-liquidity hours
# - Position sizing: 0.20 (discrete to minimize fee churn)
# - Stoploss: Signal → 0 when price closes below/above H4/L4 or opposite Camarilla level
# - Target: 15-35 trades/year (60-140 total over 4 years) to avoid fee drag
# - Works in bull/bear: Trend filter ensures we trade with higher timeframe momentum

name = "1h_4h_1d_camarilla_breakout_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h indicators
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX(14) for regime filter
    # TR calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    
    # +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr_1d, 14)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute Camarilla levels for 1d (using previous day's OHLC)
    camarilla_h3 = np.zeros(len(close_1d))
    camarilla_l3 = np.zeros(len(close_1d))
    camarilla_h4 = np.zeros(len(close_1d))
    camarilla_l4 = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        # Use previous day's OHLC to avoid look-ahead
        prev_close = close_1d[i-1]
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        range_ = prev_high - prev_low
        camarilla_h3[i] = prev_close + range_ * 1.1 / 4
        camarilla_l3[i] = prev_close - range_ * 1.1 / 4
        camarilla_h4[i] = prev_close + range_ * 1.1 / 2
        camarilla_l4[i] = prev_close - range_ * 1.1 / 2
    
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute 1h volume confirmation
    volume_1h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    vol_spike_1h = volume_1h > (1.5 * avg_volume_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0  # Close position outside session
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price closes below L4 or below H4 (profit target)
            if prices['close'].iloc[i] < camarilla_l4_aligned[i] or \
               prices['close'].iloc[i] < camarilla_h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: Price closes above H4 or above L4 (profit target)
            if prices['close'].iloc[i] > camarilla_h4_aligned[i] or \
               prices['close'].iloc[i] > camarilla_l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for breakout with trend and volume filters
            if vol_spike_1h[i] and adx_1d_aligned[i] > 25:  # Trending regime
                # Long signal: Price breaks above H3 in 4h uptrend
                if prices['close'].iloc[i] > camarilla_h3_aligned[i] and \
                   prices['close'].iloc[i] > ema_50_4h_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short signal: Price breaks below L3 in 4h downtrend
                elif prices['close'].iloc[i] < camarilla_l3_aligned[i] and \
                     prices['close'].iloc[i] < ema_50_4h_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals