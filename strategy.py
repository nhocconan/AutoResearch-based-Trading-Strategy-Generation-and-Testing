#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h trend (EMA21) and 1d volatility regime (ATR ratio) for direction,
# with 1h RSI pullback entries. Designed for low frequency (15-30 trades/year) to avoid fee drag.
# Works in bull/bear: trend filter aligns with higher timeframe, volatility regime avoids chop,
# RSI pullback provides precise entries. Uses session filter (08-20 UTC) to reduce noise.

name = "1h_ema21_atr_ratio_rsi_pullback_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-market to London close)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # 4h trend filter: EMA21
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # 1d volatility regime: ATR ratio (current ATR / 20-period average ATR)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_1d / atr_ma_20
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # 1h RSI for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available or outside session
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(rsi[i]) or not session_mask[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: price vs 4h EMA21
        uptrend = close[i] > ema_21_4h_aligned[i]
        downtrend = close[i] < ema_21_4h_aligned[i]
        
        # Volatility regime: avoid extreme volatility (ratio > 2.0) and dead zone (ratio < 0.5)
        vol_regime_ok = (atr_ratio_aligned[i] >= 0.5) & (atr_ratio_aligned[i] <= 2.0)
        
        # RSI conditions for pullback entries
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit on trend reversal or RSI overbought
            if not uptrend or rsi_overbought:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long position
        elif position == -1:  # Short position
            # Exit on trend reversal or RSI oversold
            if not downtrend or rsi_oversold:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short position
        else:  # Flat, look for entry
            # Enter long on uptrend + good volatility + RSI oversold pullback
            if uptrend and vol_regime_ok and rsi_oversold:
                position = 1
                signals[i] = 0.20
            # Enter short on downtrend + good volatility + RSI overbought pullback
            elif downtrend and vol_regime_ok and rsi_overbought:
                position = -1
                signals[i] = -0.20
    
    return signals