# BTC/ETH/ETH SOL-Specific Strategy: 6h Bollinger Width Percentile + Volume Spike + 1d Trend Filter
# HYPOTHESIS: In low volatility regimes (Bollinger Width < 20th percentile), price tends to mean-revert.
# High volatility spikes (volume > 2x 20-period average) during low-vol regimes signal exhaustion and imminent reversal.
# Trend filter (1d EMA50) ensures we only take mean-reversion trades in the direction of higher timeframe trend.
# Works in both bull/bear: mean reversion works in ranging markets, trend filter avoids counter-trend trades in strong trends.
# Target: 15-35 trades/year (60-140 over 4 years) to avoid fee drag.

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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === Daily Bollinger Bands (20, 2) for volatility regime ===
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # BB middle (SMA20)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    # BB std dev
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    # BB width = (upper - lower) / middle = (2 * std) / sma
    bb_width = (2 * bb_std) / sma_20
    # Avoid division by zero
    bb_width = np.where(sma_20 == 0, 0, bb_width)
    
    # BB width percentile (252-day lookback for regime - ~1 year of daily data)
    bb_width_pct = pd.Series(bb_width).rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # === Daily EMA50 for trend filter ===
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    # Align indicators to 6h timeframe
    bb_width_pct_aligned = align_htf_to_ltf(prices, df_1d, bb_width_pct)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(bb_width_pct_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        bb_width_pct_val = bb_width_pct_aligned[i]
        ema_trend = ema_50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter LONG in low volatility (range) + price below trend (mean reversion long) + volume spike
            if (bb_width_pct_val < 20 and  # Low volatility regime (BB width < 20th percentile)
                price_close < ema_trend and  # Price below EMA50 -> potential long mean reversion
                vol_ratio_val > 2.0):      # Volume spike > 2x average
                signals[i] = 0.25
                position = 1
            # Enter SHORT in low volatility (range) + price above trend (mean reversion short) + volume spike
            elif (bb_width_pct_val < 20 and   # Low volatility regime
                  price_close > ema_trend and  # Price above EMA50 -> potential short mean reversion
                  vol_ratio_val > 2.0):      # Volume spike > 2x average
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: volatility expansion OR price crosses EMA50 (trend resumption) OR volume normalizes
            if position == 1 and (bb_width_pct_val > 50 or price_close > ema_trend or vol_ratio_val < 1.1):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (bb_width_pct_val > 50 or price_close < ema_trend or vol_ratio_val < 1.1):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_BB_Width_Percentile_Volume_Spike_EMA50_MeanReversion"
timeframe = "6h"
leverage = 1.0