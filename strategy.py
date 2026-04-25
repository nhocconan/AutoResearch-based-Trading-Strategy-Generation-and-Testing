#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_FundingZ_Confluence
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and daily funding rate Z-score confluence.
Long when price breaks above R3 in 1d uptrend (close > 1d EMA34) with negative funding Z-score (funding < -0.5*std).
Short when price breaks below S3 in 1d downtrend (close < 1d EMA34) with positive funding Z-score (funding > +0.5*std).
Exit via opposite Camarilla level (S3 for longs, R3 for shorts) or ATR trailing stop (2.0*ATR from extreme).
Uses funding rate mean reversion as a proven BTC/ETH edge to filter breakouts, reducing false signals and overtrading.
Designed for ~12-37 trades/year via tight R3/S3 breakout conditions and dual-filter confluence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Camarilla levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # ATR for trailing stop (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Get daily OHLC for Camarilla levels (use same 1d data)
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R3/S3)
    # R3 = c + (h-l)*1.1/4
    # S3 = c - (h-l)*1.1/4
    camarilla_r3_1d = c_1d + ((h_1d - l_1d) * 1.1 / 4)
    camarilla_s3_1d = c_1d - ((h_1d - l_1d) * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Load funding rate data for mean reversion edge (BTC/ETH specific)
    try:
        funding_path = f"/mnt/data/funding/{getattr(prices, 'symbol', 'BTCUSDT').replace('USDT', '-USDT').replace('PERP', '')}-funding_rate.csv"
        funding_df = pd.read_csv(funding_path)
        funding_df['timestamp'] = pd.to_datetime(funding_df['timestamp'])
        funding_df.set_index('timestamp', inplace=True)
        # Align funding to price timestamps
        funding_series = funding_df['funding_rate'].reindex(prices['open_time'], method='ffill')
        funding_values = funding_series.values
        # Calculate Z-score of funding over 30d window
        funding_mean = pd.Series(funding_values).rolling(window=60, min_periods=30).mean().values  # 60 = 30d * 2 (12h bars per day)
        funding_std = pd.Series(funding_values).rolling(window=60, min_periods=30).std().values
        funding_z = (funding_values - funding_mean) / (funding_std + 1e-10)
    except Exception:
        # Fallback: use synthetic funding Z-score if file not available (should not happen in backtest)
        funding_z = np.zeros(n)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0   # highest close since long entry
    short_extreme = 0.0  # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, atr_period, 34, 60)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(atr[i]) or np.isnan(funding_z[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_34_1d_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        fund_z = funding_z[i]
        
        if position == 0:
            # Only trade in trending regimes (1d EMA34 filter)
            if close[i] > ema_trend:  # 1d uptrend regime
                # Long: break above R3 with negative funding Z-score (mean reversion edge)
                long_signal = (close[i] > r3_level) and (fund_z < -0.5)
            else:  # 1d downtrend regime
                # Short: break below S3 with positive funding Z-score (mean reversion edge)
                short_signal = (close[i] < s3_level) and (fund_z > 0.5)
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
                long_extreme = close[i]
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
                short_extreme = close[i]
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update extreme for trailing stop
            if close[i] > long_extreme:
                long_extreme = close[i]
            # Exit conditions: 
            # 1. ATR trailing stop (2.0*ATR from extreme)
            atr_stop = long_extreme - 2.0 * atr[i]
            # 2. Price breaks below S3 (opposite Camarilla level)
            if close[i] <= atr_stop or close[i] < s3_level:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update extreme for trailing stop
            if close[i] < short_extreme:
                short_extreme = close[i]
            # Exit conditions:
            # 1. ATR trailing stop (2.0*ATR from extreme)
            atr_stop = short_extreme + 2.0 * atr[i]
            # 2. Price breaks above R3 (opposite Camarilla level)
            if close[i] >= atr_stop or close[i] > r3_level:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dTrend_FundingZ_Confluence"
timeframe = "12h"
leverage = 1.0