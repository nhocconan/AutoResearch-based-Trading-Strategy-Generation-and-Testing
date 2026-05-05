#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4-period RSI mean reversion with 4h trend filter and 1d volume regime
# Long when: RSI(4) < 25 (oversold) AND 4h close > 4h EMA50 (uptrend) AND 1d volume > 1.2x 20-period MA (high conviction)
# Short when: RSI(4) > 75 (overbought) AND 4h close < 4h EMA50 (downtrend) AND 1d volume > 1.2x 20-period MA
# Exit when: RSI(4) crosses 50 (mean reversion complete) OR volume drops below 0.8x 20-period MA (low conviction)
# Uses short RSI for timely mean reversion entries, higher timeframes for trend and regime filtering
# Timeframe: 1h, HTF: 4h for trend, 1d for volume regime. Target: 80-120 total trades over 4 years (20-30/year) to avoid fee drag.

name = "1h_RSI4_4hTrend_1dVolRegime"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI(4) on 1h
    if len(close) >= 5:
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        
        avg_gain = pd.Series(gain).ewm(alpha=1/4, min_periods=4, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/4, min_periods=4, adjust=False).mean().values
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        rsi[np.isnan(rsi)] = 50  # neutral when undefined
    else:
        rsi = np.full(n, 50)
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    if len(close_4h) >= 50:
        ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    else:
        ema_50_4h = np.full(len(df_4h), np.nan)
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data ONCE before loop for volume regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    if len(volume_1d) >= 20:
        vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
        vol_regime_high = volume_1d > (1.2 * vol_ma_20_1d)  # high volume conviction
        vol_regime_low = volume_1d < (0.8 * vol_ma_20_1d)   # low volume conviction (exit condition)
    else:
        vol_regime_high = np.zeros(len(df_1d), dtype=bool)
        vol_regime_low = np.zeros(len(df_1d), dtype=bool)
    
    # Align 1d volume regime to 1h timeframe
    vol_regime_high_aligned = align_htf_to_ltf(prices, df_1d, vol_regime_high.astype(float))
    vol_regime_low_aligned = align_htf_to_ltf(prices, df_1d, vol_regime_low.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any value is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_regime_high_aligned[i]) or np.isnan(vol_regime_low_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: RSI oversold + 4h uptrend + high volume conviction
            if (rsi[i] < 25 and 
                close[i] > ema_50_4h_aligned[i] and 
                vol_regime_high_aligned[i] == 1.0):
                signals[i] = 0.20
                position = 1
            # Short conditions: RSI overbought + 4h downtrend + high volume conviction
            elif (rsi[i] > 75 and 
                  close[i] < ema_50_4h_aligned[i] and 
                  vol_regime_high_aligned[i] == 1.0):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI mean reversion (cross above 50) OR low volume conviction
            if (rsi[i] > 50 or vol_regime_low_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI mean reversion (cross below 50) OR low volume conviction
            if (rsi[i] < 50 or vol_regime_low_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals