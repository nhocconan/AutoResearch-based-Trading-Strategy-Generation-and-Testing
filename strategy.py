#!/usr/bin/env python3
name = "6h_Volatility_Regime_Adaptive_Momentum"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for regime detection and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR for regime detection (choppy vs trending)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation for 1d
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR percentile rank (20-period) for regime detection
    atr_rank = pd.Series(atr_1d).rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 20 else np.nan, raw=False
    ).values
    atr_rank_aligned = align_htf_to_ltf(prices, df_1d, atr_rank)
    
    # Calculate 1d ADX for trend strength
    # +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # True Range (already calculated as tr_1d)
    tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / tr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 60-period EMA for 6h trend filter
    close_series = pd.Series(close)
    ema_60 = close_series.ewm(span=60, min_periods=60).mean().values
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_rank_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(ema_60[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Regime-based logic:
        # Choppy market (high ATR rank): mean reversion at extremes
        # Trending market (low ATR rank, high ADX): momentum continuation
        
        if position == 0:
            # Choppy regime: ATR rank > 0.7 (high volatility, likely ranging)
            if atr_rank_aligned[i] > 0.7:
                # Mean reversion: look for extreme moves combined with volume
                # Calculate 6-period RSI for short-term extremes
                if i >= 6:
                    rsi_gain = np.where(np.diff(close[i-5:i+1]) > 0, np.diff(close[i-5:i+1]), 0)
                    rsi_loss = np.where(np.diff(close[i-5:i+1]) < 0, -np.diff(close[i-5:i+1]), 0)
                    avg_gain = np.mean(rsi_gain) if len(rsi_gain) > 0 else 0
                    avg_loss = np.mean(rsi_loss) if len(rsi_loss) > 0 else 0
                    if avg_loss == 0:
                        rsi = 100
                    elif avg_gain == 0:
                        rsi = 0
                    else:
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                    
                    # Long when oversold (RSI < 30) with volume
                    # Short when overbought (RSI > 70) with volume
                    if rsi < 30 and volume_filter[i]:
                        signals[i] = 0.25
                        position = 1
                    elif rsi > 70 and volume_filter[i]:
                        signals[i] = -0.25
                        position = -1
            # Trending regime: ATR rank <= 0.5 AND ADX > 25
            elif atr_rank_aligned[i] <= 0.5 and adx_1d_aligned[i] > 25:
                # Trend continuation: buy dips in uptrend, sell rallies in downtrend
                if close[i] > ema_60[i] and volume_filter[i]:
                    # Uptrend: buy on pullbacks
                    if i > 0 and close[i] < close[i-1]:  # Pullback condition
                        signals[i] = 0.25
                        position = 1
                elif close[i] < ema_60[i] and volume_filter[i]:
                    # Downtrend: sell on bounces
                    if i > 0 and close[i] > close[i-1]:  # Bounce condition
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit conditions
            if atr_rank_aligned[i] > 0.7:
                # In choppy mode, exit when RSI returns to neutral (50)
                if i >= 6:
                    rsi_gain = np.where(np.diff(close[i-5:i+1]) > 0, np.diff(close[i-5:i+1]), 0)
                    rsi_loss = np.where(np.diff(close[i-5:i+1]) < 0, -np.diff(close[i-5:i+1]), 0)
                    avg_gain = np.mean(rsi_gain) if len(rsi_gain) > 0 else 0
                    avg_loss = np.mean(rsi_loss) if len(rsi_loss) > 0 else 0
                    if avg_loss == 0:
                        rsi = 100
                    elif avg_gain == 0:
                        rsi = 0
                    else:
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                    if rsi >= 50:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                else:
                    signals[i] = 0.25
            else:
                # In trending mode, exit when price crosses below EMA60
                if close[i] < ema_60[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit conditions
            if atr_rank_aligned[i] > 0.7:
                # In choppy mode, exit when RSI returns to neutral (50)
                if i >= 6:
                    rsi_gain = np.where(np.diff(close[i-5:i+1]) > 0, np.diff(close[i-5:i+1]), 0)
                    rsi_loss = np.where(np.diff(close[i-5:i+1]) < 0, -np.diff(close[i-5:i+1]), 0)
                    avg_gain = np.mean(rsi_gain) if len(rsi_gain) > 0 else 0
                    avg_loss = np.mean(rsi_loss) if len(rsi_loss) > 0 else 0
                    if avg_loss == 0:
                        rsi = 100
                    elif avg_gain == 0:
                        rsi = 0
                    else:
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                    if rsi <= 50:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                # In trending mode, exit when price crosses above EMA60
                if close[i] > ema_60[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals