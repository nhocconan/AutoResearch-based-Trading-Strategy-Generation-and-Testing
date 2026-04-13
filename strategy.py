#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 1d KAMA direction + RSI(14) + 1w choppiness regime filter
    # Designed for low trade frequency (7-25/year) to minimize fee drag
    # Works in bull/bear markets: KAMA catches trend, RSI avoids extremes, chop filter avoids whipsaws
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 1d KAMA (adaptive trend)
    # Efficiency ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1w data for choppiness regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w choppiness index
    atr_1w = np.zeros_like(high_1w)
    tr1 = np.abs(np.diff(high_1w, prepend=high_1w[0]))
    tr2 = np.abs(np.diff(low_1w, prepend=low_1w[0]))
    tr3 = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    chop = np.divide(
        atr_1w * np.sqrt(14),
        np.maximum(max_high - min_low, 1e-10),
        out=np.zeros_like(atr_1w),
        where=(max_high - min_low)!=0
    )
    chop *= 100
    
    # Align all HTF indicators to 1d primary timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction: price relative to KAMA
        above_kama = close[i] > kama_aligned[i]
        below_kama = close[i] < kama_aligned[i]
        
        # RSI conditions: avoid extremes, look for mean reversion
        rsi_not_overbought = rsi_aligned[i] < 70
        rsi_not_oversold = rsi_aligned[i] > 30
        rsi_bullish = rsi_aligned[i] > 50
        rsi_bearish = rsi_aligned[i] < 50
        
        # Choppiness regime: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trending (trend follow)
        chop_high = chop_aligned[i] > 61.8  # ranging market
        chop_low = chop_aligned[i] < 38.2   # trending market
        
        # Entry conditions
        # Long: price above KAMA (uptrend) + RSI not overbought + in ranging market (mean reversion)
        enter_long = above_kama and rsi_not_overbought and chop_high and rsi_bullish
        # Short: price below KAMA (downtrend) + RSI not oversold + in ranging market (mean reversion)
        enter_short = below_kama and rsi_not_oversold and chop_high and rsi_bearish
        
        # Exit conditions: opposite signal or RSI extreme
        exit_long = (below_kama or rsi_aligned[i] >= 70) and position == 1
        exit_short = (above_kama or rsi_aligned[i] <= 30) and position == -1
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_kama_rsi_chop_regime_v1"
timeframe = "1d"
leverage = 1.0