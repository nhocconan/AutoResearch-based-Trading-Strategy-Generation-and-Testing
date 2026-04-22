#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and 1d volume regime
# Strategy trades pullbacks in the direction of 4h EMA(50) trend during high-volume regimes
# Uses 1h RSI(14) for entry timing: long when RSI < 30 in uptrend, short when RSI > 70 in downtrend
# Volume filter: only trade when 1h volume > 1.5x 20-period average to avoid low-activity periods
# Designed for 1h timeframe with disciplined entry to avoid overtrading (target: 15-30 trades/year)
# Works in bull markets via trend-following pullbacks and in bear markets via counter-trend bounces
# Uses discrete position sizing (0.20) to minimize fee churn

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for trend filter (ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    # 1d volume 20-period average for regime detection
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # 1h RSI(14) for entry timing
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # 1h volume 20-period average for volume spike filter
    vol_avg_20_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_avg_20_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume regime: only trade when 1h volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_avg_20_1h[i]
        
        if position == 0 and volume_filter:
            # Long: 4h uptrend + RSI oversold (pullback entry)
            if close[i] > ema_50_4h_aligned[i] and rsi[i] < 30:
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend + RSI overbought (pullback entry)
            elif close[i] < ema_50_4h_aligned[i] and rsi[i] > 70:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: RSI overbought or trend breakdown
                if rsi[i] > 70 or close[i] < ema_50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                # Exit short: RSI oversold or trend reversal
                if rsi[i] < 30 or close[i] > ema_50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_EMA50_RSI_Pullback_VolumeRegime"
timeframe = "1h"
leverage = 1.0