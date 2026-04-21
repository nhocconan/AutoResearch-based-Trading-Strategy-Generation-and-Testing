#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # === Daily Bollinger Bands for volatility regime ===
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    # BB Width for regime detection
    bb_width = ((upper_bb - lower_bb) / sma_20) * 100
    bb_width_pct = pd.Series(bb_width).rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    bb_width_pct_aligned = align_htf_to_ltf(prices, df_1d, bb_width_pct)
    
    # === Weekly EMA34 for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 100:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily volume confirmation ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(bb_width_pct_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        bb_width_val = bb_width_pct_aligned[i]
        weekly_trend = ema_34_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long in low volatility (range) + weekly uptrend + volume
            if (bb_width_val < 30 and  # Low volatility regime (range)
                price_close > weekly_trend and
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short in low volatility (range) + weekly downtrend + volume
            elif (bb_width_val < 30 and   # Low volatility regime (range)
                  price_close < weekly_trend and
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when volatility increases (trending regime) or opposite condition
            if position == 1 and (bb_width_val > 70 or price_close < weekly_trend):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (bb_width_val > 70 or price_close > weekly_trend):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_BB_Width_Regime_WeeklyEMA34_Trend_Volume"
timeframe = "1d"
leverage = 1.0