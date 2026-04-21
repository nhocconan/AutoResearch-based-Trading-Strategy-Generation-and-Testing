#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Weekly ATR for volatility regime ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR(10)
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # ATR(10) percentile (52-week lookback for regime)
    atr_percentile = pd.Series(atr_10).rolling(window=52, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Align ATR percentile to daily timeframe
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1w, atr_percentile)
    
    # === Weekly close for trend filter ===
    weekly_close = close_1w
    weekly_ema21 = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_ema21_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema21)
    
    # === Daily volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(weekly_ema21_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        atr_percentile_val = atr_percentile_aligned[i]
        weekly_trend = weekly_ema21_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long in low volatility (range) + weekly uptrend + volume
            if (atr_percentile_val < 30 and  # Low volatility regime
                price_close > weekly_trend and
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short in low volatility (range) + weekly downtrend + volume
            elif (atr_percentile_val < 30 and   # Low volatility regime
                  price_close < weekly_trend and
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when volatility increases (trending regime) or opposite condition
            if position == 1 and (atr_percentile_val > 70 or price_close < weekly_trend):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (atr_percentile_val > 70 or price_close > weekly_trend):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_ATR_Volatility_Regime_WeeklyEMA21_Trend_Volume"
timeframe = "1d"
leverage = 1.0