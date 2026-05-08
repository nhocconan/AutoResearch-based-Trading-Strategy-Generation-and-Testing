#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h EMA(21)/EMA(50) cross with 1-week RSI(14) filter and volume confirmation
# EMA cross provides trend direction, weekly RSI prevents entries in overbought/oversold conditions
# Volume spike confirms institutional participation. Targets 20-40 trades per year.
# Works in bull markets via trend following, in bear markets via avoiding counter-trend entries.
# Uses discrete position sizing (0.25) to minimize fee churn.

name = "6h_EMA_Cross_1wRSI_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA indicators
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly RSI for filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_conf = volume > (vol_ma * 1.8)
    
    # Align weekly RSI to 6h timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema21[i]) or np.isnan(ema50[i]) or 
            np.isnan(rsi_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema21_val = ema21[i]
        ema50_val = ema50[i]
        rsi_val = rsi_1w_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: EMA21 crosses above EMA50, RSI not overbought (<70), volume confirmation
            if ema21_val > ema50_val and ema21[i-1] <= ema50[i-1] and rsi_val < 70 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: EMA21 crosses below EMA50, RSI not oversold (>30), volume confirmation
            elif ema21_val < ema50_val and ema21[i-1] >= ema50[i-1] and rsi_val > 30 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: EMA21 crosses below EMA50 or RSI overbought (>70)
            if ema21_val < ema50_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: EMA21 crosses above EMA50 or RSI oversold (<30)
            if ema21_val > ema50_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals